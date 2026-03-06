from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TargetConfig:
    num_bins: int = 201
    c_fail: float = 200.0
    normalize_mode: str = "per_task_max_length"
    max_return_steps: int = 200



def parse_success(episode_meta: dict, success_key: str) -> bool:
    if success_key and success_key in episode_meta:
        value = episode_meta[success_key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "success", "successful"}
    trajectory_type = str(episode_meta.get("trajectory_type", "")).lower()
    if trajectory_type:
        return "success" in trajectory_type
    return False



def terminal_reward(episode_meta: dict, success: bool, reward_key: str, c_fail: float) -> float:
    if reward_key and reward_key in episode_meta:
        return float(episode_meta[reward_key])
    return 0.0 if success else -float(c_fail)



def compute_episode_returns(length: int, term_reward: float) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    returns = np.zeros(length, dtype=np.float32)
    returns[-1] = np.float32(term_reward)
    for idx in range(length - 2, -1, -1):
        returns[idx] = returns[idx + 1] - 1.0
    return returns



def normalize_returns(
    returns: np.ndarray,
    scale: float,
) -> np.ndarray:
    if scale <= 0:
        scale = 1.0
    normalized = returns / scale
    return np.clip(normalized, -1.0, 0.0).astype(np.float32)



def normalized_to_bin(values: np.ndarray, num_bins: int) -> np.ndarray:
    values = np.clip(values, -1.0, 0.0)
    scaled = (values + 1.0) * (num_bins - 1)
    return np.rint(scaled).astype(np.int64)



def bin_to_normalized(bin_indices: np.ndarray, num_bins: int) -> np.ndarray:
    return (bin_indices.astype(np.float32) / float(num_bins - 1)) - 1.0



def compute_task_scales(task_to_lengths: dict[str, list[int]], max_return_steps: int) -> dict[str, float]:
    scales: dict[str, float] = {}
    for task, lengths in task_to_lengths.items():
        max_len = max(lengths) if lengths else 1
        scales[task] = float(max(1, min(max_len, max_return_steps)))
    return scales



def target_histogram(target_bins: Iterable[int], num_bins: int) -> np.ndarray:
    hist = np.zeros(num_bins, dtype=np.int64)
    for target in target_bins:
        if 0 <= target < num_bins:
            hist[target] += 1
    return hist
