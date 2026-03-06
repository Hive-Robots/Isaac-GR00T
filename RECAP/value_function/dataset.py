from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig

from .targets import (
    TargetConfig,
    compute_episode_returns,
    compute_task_scales,
    normalize_returns,
    normalized_to_bin,
    parse_success,
    terminal_reward,
)


@dataclass
class StepRecord:
    episode_local_idx: int
    step_idx: int
    task: str
    target_value: float
    target_bin: int



def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]



def _task_from_meta(meta: dict[str, Any], task_key: str) -> str:
    if task_key in meta:
        value = meta[task_key]
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)
    tasks = meta.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return ""



class RECAPValueDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        image_keys: list[str],
        language_key: str,
        task_key: str,
        success_key: str,
        reward_key: str,
        target_cfg: TargetConfig,
        split: str,
        train_split: float,
        seed: int,
        max_episode_cache: int = 8,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.image_keys = image_keys
        self.language_key = language_key
        self.task_key = task_key
        self.success_key = success_key
        self.reward_key = reward_key
        self.target_cfg = target_cfg
        self.split = split
        self.max_episode_cache = max_episode_cache

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {self.dataset_path}")

        modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=image_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[language_key]),
        }
        self.loader = LeRobotEpisodeLoader(
            dataset_path=self.dataset_path,
            modality_configs=modality_configs,
            video_backend="torchcodec",
            video_backend_kwargs=None,
        )

        self.episodes_meta = self.loader.episodes_metadata
        self.records: list[StepRecord] = self._build_records()

        rng = random.Random(seed)
        episode_ids = list(range(len(self.episodes_meta)))
        rng.shuffle(episode_ids)
        split_idx = int(len(episode_ids) * train_split)
        selected = set(episode_ids[:split_idx] if split == "train" else episode_ids[split_idx:])
        self.records = [record for record in self.records if record.episode_local_idx in selected]

        self._episode_cache: dict[int, Any] = {}
        self._episode_cache_order: list[int] = []

    def _build_records(self) -> list[StepRecord]:
        task_lengths: dict[str, list[int]] = {}
        per_episode_task: list[str] = []

        for meta in self.episodes_meta:
            task = _task_from_meta(meta, self.task_key)
            per_episode_task.append(task)
            length = int(meta["length"])
            task_lengths.setdefault(task, []).append(length)

        task_scales = compute_task_scales(task_lengths, self.target_cfg.max_return_steps)

        records: list[StepRecord] = []
        for ep_idx, meta in enumerate(self.episodes_meta):
            length = int(meta["length"])
            task = per_episode_task[ep_idx]
            success = parse_success(meta, self.success_key)
            term_r = terminal_reward(meta, success, self.reward_key, self.target_cfg.c_fail)

            returns = compute_episode_returns(length=length, term_reward=term_r)
            scale = task_scales.get(task, float(self.target_cfg.max_return_steps))
            norm = normalize_returns(returns, scale=scale)
            bins = normalized_to_bin(norm, self.target_cfg.num_bins)

            for step_idx in range(length):
                records.append(
                    StepRecord(
                        episode_local_idx=ep_idx,
                        step_idx=step_idx,
                        task=task,
                        target_value=float(norm[step_idx]),
                        target_bin=int(bins[step_idx]),
                    )
                )
        return records

    def _get_episode_df(self, ep_idx: int):
        if ep_idx in self._episode_cache:
            return self._episode_cache[ep_idx]

        df = self.loader[ep_idx]
        self._episode_cache[ep_idx] = df
        self._episode_cache_order.append(ep_idx)
        if len(self._episode_cache_order) > self.max_episode_cache:
            oldest = self._episode_cache_order.pop(0)
            self._episode_cache.pop(oldest, None)
        return df

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        df = self._get_episode_df(record.episode_local_idx)
        if len(df) == 0:
            fallback_images = [Image.new("RGB", (224, 224), color="black") for _ in self.image_keys]
            return {
                "images": fallback_images,
                "text": record.task,
                "task": record.task,
                "target_value": record.target_value,
                "target_bin": record.target_bin,
            }

        step_idx = min(record.step_idx, len(df) - 1)

        images: list[Image.Image] = []
        for key in self.image_keys:
            pil = df[f"video.{key}"].iloc[step_idx]
            if not isinstance(pil, Image.Image):
                pil = Image.fromarray(np.asarray(pil))
            images.append(pil)

        lang_col = f"language.{self.language_key}"
        if lang_col in df.columns:
            text = str(df[lang_col].iloc[step_idx])
        else:
            text = record.task

        return {
            "images": images,
            "text": text,
            "task": record.task,
            "target_value": record.target_value,
            "target_bin": record.target_bin,
        }



def collate_value_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    target_bins = torch.tensor([item["target_bin"] for item in batch], dtype=torch.long)
    target_values = torch.tensor([item["target_value"] for item in batch], dtype=torch.float32)
    return {
        "images": [item["images"] for item in batch],
        "texts": [item["text"] for item in batch],
        "tasks": [item["task"] for item in batch],
        "target_bins": target_bins,
        "target_values": target_values,
    }
