from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class VFTrainConfig:
    dataset_path: str
    image_keys: list[str] = field(default_factory=lambda: ["top"])
    language_key: str = "task"
    success_key: str = "success"
    reward_key: str = ""
    task_key: str = "tasks"

    num_bins: int = 201
    c_fail: float = 200.0
    normalize_mode: str = "per_task_max_length"
    max_return_steps: int = 200

    value_hidden_dim: int = 1024
    freeze_encoder: bool = True
    dtype: str = "bfloat16"
    encoder_model_path: str = "nvidia/Eagle-Block2A-2B-v2"
    encoder_backend: str = "eagle"

    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_steps: int = 10_000
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 50

    device: str = "cuda"
    num_workers: int = 2
    seed: int = 42
    output_dir: str = "RECAP/value_function/outputs"
    resume: str = ""

    train_split: float = 0.9
    val_max_batches: int = 100
    inference_only: bool = False



def parse_args() -> VFTrainConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Train RECAP value function")

    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--image-keys", type=str, default="top")
    parser.add_argument("--language-key", type=str, default="task")
    parser.add_argument("--success-key", type=str, default="success")
    parser.add_argument("--reward-key", type=str, default="")
    parser.add_argument("--task-key", type=str, default="tasks")

    parser.add_argument("--num-bins", type=int, default=201)
    parser.add_argument("--c-fail", type=float, default=200.0)
    parser.add_argument("--normalize-mode", type=str, default="per_task_max_length")
    parser.add_argument("--max-return-steps", type=int, default=200)

    parser.add_argument("--value-hidden-dim", type=int, default=1024)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--encoder-model-path", type=str, default="nvidia/Eagle-Block2A-2B-v2")
    parser.add_argument("--encoder-backend", type=str, default="eagle", choices=["eagle", "mock"])

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="RECAP/value_function/outputs")
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--val-max-batches", type=int, default=100)
    parser.add_argument("--inference-only", action="store_true", default=False)

    args = parser.parse_args()

    return VFTrainConfig(
        dataset_path=args.dataset_path,
        image_keys=_split_csv(args.image_keys),
        language_key=args.language_key,
        success_key=args.success_key,
        reward_key=args.reward_key,
        task_key=args.task_key,
        num_bins=args.num_bins,
        c_fail=args.c_fail,
        normalize_mode=args.normalize_mode,
        max_return_steps=args.max_return_steps,
        value_hidden_dim=args.value_hidden_dim,
        freeze_encoder=args.freeze_encoder,
        dtype=args.dtype,
        encoder_model_path=args.encoder_model_path,
        encoder_backend=args.encoder_backend,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        train_split=args.train_split,
        val_max_batches=args.val_max_batches,
        inference_only=args.inference_only,
    )



def to_dict(config: VFTrainConfig) -> dict[str, Any]:
    return asdict(config)



def save_config(config: VFTrainConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(config), f, indent=2)
