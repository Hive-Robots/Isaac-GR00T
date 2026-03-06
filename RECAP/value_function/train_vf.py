from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from value_function.config import VFTrainConfig, parse_args, save_config
    from value_function.dataset import RECAPValueDataset, collate_value_batch
    from value_function.encoder import build_encoder
    from value_function.engine import (
        MetricTracker,
        evaluate,
        load_checkpoint,
        save_checkpoint,
        step_forward,
    )
    from value_function.model import ValueFunctionHead
    from value_function.targets import TargetConfig
else:
    from .config import VFTrainConfig, parse_args, save_config
    from .dataset import RECAPValueDataset, collate_value_batch
    from .encoder import build_encoder
    from .engine import MetricTracker, evaluate, load_checkpoint, save_checkpoint, step_forward
    from .model import ValueFunctionHead
    from .targets import TargetConfig


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)



def _create_dataloaders(config: VFTrainConfig) -> tuple[DataLoader, DataLoader, RECAPValueDataset, RECAPValueDataset]:
    target_cfg = TargetConfig(
        num_bins=config.num_bins,
        c_fail=config.c_fail,
        normalize_mode=config.normalize_mode,
        max_return_steps=config.max_return_steps,
    )

    train_ds = RECAPValueDataset(
        dataset_path=config.dataset_path,
        image_keys=config.image_keys,
        language_key=config.language_key,
        task_key=config.task_key,
        success_key=config.success_key,
        reward_key=config.reward_key,
        target_cfg=target_cfg,
        split="train",
        train_split=config.train_split,
        seed=config.seed,
    )
    val_ds = RECAPValueDataset(
        dataset_path=config.dataset_path,
        image_keys=config.image_keys,
        language_key=config.language_key,
        task_key=config.task_key,
        success_key=config.success_key,
        reward_key=config.reward_key,
        target_cfg=target_cfg,
        split="val",
        train_split=config.train_split,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_value_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_value_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds, val_ds



def _log_metrics(prefix: str, step: int, metrics: dict) -> None:
    hist = np.array(metrics["target_histogram"], dtype=np.int64)
    nonzero = np.where(hist > 0)[0]
    hist_summary = [(int(idx), int(hist[idx])) for idx in nonzero[:10]]
    logging.info(
        "%s step=%d loss=%.5f top1=%.4f top5=%.4f mae=%.5f hist_head=%s",
        prefix,
        step,
        metrics["loss"],
        metrics["top1_acc"],
        metrics["top5_acc"],
        metrics["expected_value_mae"],
        hist_summary,
    )



def run(config: VFTrainConfig) -> None:
    _setup_logging()
    _set_seed(config.seed)

    device = _resolve_device(config.device)
    logging.info("Using device=%s", device)

    train_loader, val_loader, train_ds, val_ds = _create_dataloaders(config)
    logging.info("Train samples=%d, Val samples=%d", len(train_ds), len(val_ds))

    encoder = build_encoder(
        backend=config.encoder_backend,
        model_path=config.encoder_model_path,
        dtype=config.dtype,
        freeze_encoder=config.freeze_encoder,
    ).to(device)

    head = ValueFunctionHead(
        embed_dim=encoder.embedding_dim,
        hidden_dim=config.value_hidden_dim,
        num_bins=config.num_bins,
    ).to(device)

    params = list(head.parameters())
    if not config.freeze_encoder:
        params += [p for p in encoder.parameters() if p.requires_grad]

    optimizer = AdamW(params=params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.max_steps, 1))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")

    global_step = 0
    if config.resume:
        global_step = load_checkpoint(config.resume, head=head, encoder=encoder, optimizer=optimizer, scheduler=scheduler)
        logging.info("Resumed from %s at step=%d", config.resume, global_step)

    if config.inference_only:
        batch = next(iter(val_loader))
        head.eval()
        encoder.eval()
        with torch.no_grad():
            embedding = encoder.encode_batch(batch["images"], batch["texts"], device)
            logits = head(embedding)
            probs = torch.softmax(logits, dim=-1)
            bin_values = torch.linspace(-1.0, 0.0, steps=config.num_bins, device=device)
            expected_value = (probs * bin_values.unsqueeze(0)).sum(dim=-1)
        logging.info("Inference-only expected values: %s", expected_value[:8].detach().cpu().tolist())
        return

    tracker = MetricTracker(num_bins=config.num_bins)
    train_iter = itertools.cycle(train_loader)

    head.train()
    encoder.train(not config.freeze_encoder)

    while global_step < config.max_steps:
        batch = next(train_iter)

        logits, target_bins, target_values, loss = step_forward(
            encoder=encoder,
            head=head,
            batch=batch,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        tracker.update(logits.detach(), target_bins.detach(), target_values.detach(), float(loss.item()))
        global_step += 1

        if global_step % config.log_every == 0:
            metrics = tracker.compute()
            _log_metrics("train", global_step, metrics)
            tracker.reset()

        if global_step % config.eval_every == 0:
            eval_metrics = evaluate(
                encoder=encoder,
                head=head,
                dataloader=val_loader,
                device=device,
                num_bins=config.num_bins,
                max_batches=config.val_max_batches,
            )
            _log_metrics("eval", global_step, eval_metrics)
            head.train()
            encoder.train(not config.freeze_encoder)

        if global_step % config.save_every == 0:
            ckpt_path = save_checkpoint(
                output_dir=str(output_dir),
                step=global_step,
                config=config,
                head=head,
                encoder=encoder,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            logging.info("Saved checkpoint: %s", ckpt_path)

    final_ckpt = save_checkpoint(
        output_dir=str(output_dir),
        step=global_step,
        config=config,
        head=head,
        encoder=encoder,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    logging.info("Training finished at step=%d, final checkpoint=%s", global_step, final_ckpt)



def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
