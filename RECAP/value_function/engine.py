from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import VFTrainConfig


class MetricTracker:
    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.count = 0
        self.top1_correct = 0
        self.top5_correct = 0
        self.mae_sum = 0.0
        self.hist = np.zeros(self.num_bins, dtype=np.int64)

    def update(self, logits: torch.Tensor, target_bins: torch.Tensor, target_values: torch.Tensor, loss: float):
        bsz = target_bins.shape[0]
        self.loss_sum += float(loss) * bsz
        self.count += bsz

        top1 = logits.argmax(dim=-1)
        self.top1_correct += int((top1 == target_bins).sum().item())

        k = min(5, logits.shape[-1])
        topk = torch.topk(logits, k=k, dim=-1).indices
        top5_match = (topk == target_bins.unsqueeze(-1)).any(dim=-1)
        self.top5_correct += int(top5_match.sum().item())

        probs = torch.softmax(logits, dim=-1)
        bin_values = torch.linspace(-1.0, 0.0, steps=logits.shape[-1], device=logits.device)
        expected_value = (probs * bin_values.unsqueeze(0)).sum(dim=-1)
        self.mae_sum += float(torch.abs(expected_value - target_values).sum().item())

        bincount = torch.bincount(target_bins.detach().cpu(), minlength=self.num_bins).numpy()
        self.hist += bincount

    def compute(self) -> dict[str, Any]:
        denom = max(self.count, 1)
        return {
            "loss": self.loss_sum / denom,
            "top1_acc": self.top1_correct / denom,
            "top5_acc": self.top5_correct / denom,
            "expected_value_mae": self.mae_sum / denom,
            "target_histogram": self.hist.tolist(),
        }



def step_forward(
    encoder,
    head,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_bins = batch["target_bins"].to(device)
    target_values = batch["target_values"].to(device)

    embedding = encoder.encode_batch(
        images=batch["images"],
        texts=batch["texts"],
        device=device,
    )
    logits = head(embedding)
    loss = F.cross_entropy(logits, target_bins)
    return logits, target_bins, target_values, loss



def evaluate(
    encoder,
    head,
    dataloader,
    device: torch.device,
    num_bins: int,
    max_batches: int,
) -> dict[str, Any]:
    head.eval()
    tracker = MetricTracker(num_bins=num_bins)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            logits, target_bins, target_values, loss = step_forward(
                encoder=encoder,
                head=head,
                batch=batch,
                device=device,
            )
            tracker.update(logits, target_bins, target_values, float(loss.item()))
            if (batch_idx + 1) >= max_batches:
                break

    return tracker.compute()



def save_checkpoint(
    output_dir: str,
    step: int,
    config: VFTrainConfig,
    head,
    encoder,
    optimizer,
    scheduler,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_step_{step}.pt"

    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "head": head.state_dict(),
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        ckpt_path,
    )
    return str(ckpt_path)



def load_checkpoint(path: str, head, encoder, optimizer=None, scheduler=None) -> int:
    state = torch.load(path, map_location="cpu")
    head.load_state_dict(state["head"])
    encoder.load_state_dict(state["encoder"], strict=False)

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    return int(state.get("step", 0))
