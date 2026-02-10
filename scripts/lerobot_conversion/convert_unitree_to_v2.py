"""Utilities to convert a Unitree XR_teleoperate dataset into LeRobot v2.1.

This script writes v2.1 datasets directly (no v3 intermediate). It currently
supports `mode="video"` only.

Usage example
-------------

python scripts/lerobot_conversion/convert_unitree_to_v2.py \
    --raw-dir /path/to/xr_teleoperate_dataset \
    --repo-id your_name/your_dataset \
    --robot-type Unitree_G1_Dex3 \
    --mode video \
    --fps 25

Custom modality config example (overrides hardcoded robot config):

python scripts/lerobot_conversion/convert_unitree_to_v2.py \
    --raw-dir /path/to/xr_teleoperate_dataset \
    --repo-id your_name/your_dataset \
    --robot-type Unitree_G1_Dex3 \
    --modality-config-path /path/to/modality.json \
    --mode video \
    --fps 25
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

from huggingface_hub import HfApi
from lerobot.utils.constants import HF_LEROBOT_HOME
from scripts.lerobot_conversion.constants import ROBOT_CONFIGS, RobotConfig

def _get_robot_config(robot_type: str) -> RobotConfig:
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(
            f"Unsupported robot_type '{robot_type}'. Available options: "
            f"{', '.join(sorted(ROBOT_CONFIGS.keys()))}."
        )
    return ROBOT_CONFIGS[robot_type]


def _load_modality_config(modality_config_path: Path) -> tuple[RobotConfig, dict]:
    with modality_config_path.open("r", encoding="utf-8") as f:
        modality = json.load(f)

    if "state" not in modality or "action" not in modality:
        raise ValueError("modality config must include 'state' and 'action' sections.")

    state_keys = list(modality["state"].keys())
    action_keys = list(modality["action"].keys())

    if not state_keys or not action_keys:
        raise ValueError("modality config must contain non-empty 'state' and 'action' keys.")

    cameras = list(modality.get("video", {}).keys())
    camera_to_image_key = modality.get("camera_to_image_key", {})
    if not camera_to_image_key:
        if len(cameras) == 1:
            camera_to_image_key = {"color_0": cameras[0]}
        elif len(cameras) > 1:
            raise ValueError(
                "Multiple cameras detected in modality config but no 'camera_to_image_key' "
                "mapping provided. Add a top-level 'camera_to_image_key' field."
            )

    json_state_data_name = modality.get(
        "json_state_data_name", [f"{key}.qpos" for key in state_keys]
    )
    json_action_data_name = modality.get(
        "json_action_data_name", [f"{key}.qpos" for key in action_keys]
    )

    robot_config = RobotConfig(
        motors=state_keys,
        cameras=cameras,
        camera_to_image_key=camera_to_image_key,
        json_state_data_name=json_state_data_name,
        json_action_data_name=json_action_data_name,
    )
    return robot_config, modality


@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(
        self,
        data_dirs: Path,
        json_state_data_name: list[str],
        json_action_data_name: list[str],
        camera_to_image_key: dict[str, str],
    ) -> None:
        assert data_dirs is not None, "Data directory cannot be None"
        self.data_dirs = Path(data_dirs)
        self.json_file = "data.json"

        self._init_paths()
        self._init_cache()
        self.json_state_data_name = json_state_data_name
        self.json_action_data_name = json_action_data_name
        self.camera_to_image_key = camera_to_image_key

    def _init_paths(self) -> None:
        self.episode_paths = []
        for root, _dirs, files in os.walk(self.data_dirs):
            if self.json_file in files:
                self.episode_paths.append(root)

        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        return len(self.episode_paths)

    def _init_cache(self) -> list:
        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")
        return self.episodes_data_cached

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if (
                        qpos is None
                        and key_part in sample_data[key]
                        and sample_data[key][key_part] is not None
                    ):
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def get_item(self, index: int | None = None) -> dict:
        if index is None:
            file_path = np.random.choice(self.episode_paths)
            episode_idx = self.episode_paths.index(file_path)
        else:
            file_path = self.episode_paths[index]
            episode_idx = index

        episode_data = self.episodes_data_cached[episode_idx]

        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        task = episode_data.get("text", {}).get("goal", "")
        cameras = self._parse_images(file_path, episode_data)

        cam_height, cam_width = next(
            img for imgs in cameras.values() if imgs for img in imgs
        ).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": episode_idx,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
        }


def _ensure_empty_dataset_root(repo_id: str) -> Path:
    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    return dataset_path


def _write_video(
    frames: list[np.ndarray],
    out_path: Path,
    fps: float,
) -> dict:
    if not frames:
        raise ValueError(f"No frames provided for video {out_path}")
    height, width = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All frames must have the same resolution")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    return {
        "video.height": height,
        "video.width": width,
        "video.codec": "mp4v",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }


def _write_parquet(
    out_path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    episode_index: int,
    global_indices: np.ndarray,
    task_index: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "observation.state": [row.tolist() for row in state],
        "action": [row.tolist() for row in action],
        "timestamp": timestamps.astype(np.float32).tolist(),
        "frame_index": frame_indices.astype(np.int64).tolist(),
        "episode_index": [int(episode_index)] * len(state),
        "index": global_indices.astype(np.int64).tolist(),
        "task_index": [int(task_index)] * len(state),
    }

    table = pa.table(data)
    pq.write_table(table, out_path)


def _write_tasks(tasks: list[str], meta_dir: Path) -> None:
    tasks_path = meta_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")


def _write_episodes(episode_lengths: list[int], episode_tasks: list[str], meta_dir: Path) -> None:
    episodes_path = meta_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as f:
        for idx, (length, task) in enumerate(zip(episode_lengths, episode_tasks, strict=True)):
            f.write(json.dumps({"episode_index": idx, "tasks": [task], "length": length}) + "\n")


def _write_modality(
    motors: list[str],
    cameras: list[str],
    meta_dir: Path,
    *,
    modality_override: dict | None = None,
) -> None:
    modality = modality_override
    if modality is None:
        modality = {
            "state": {name: {"start": i, "end": i + 1} for i, name in enumerate(motors)},
            "action": {name: {"start": i, "end": i + 1} for i, name in enumerate(motors)},
            "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in cameras},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        }
    modality_path = meta_dir / "modality.json"
    with modality_path.open("w", encoding="utf-8") as f:
        json.dump(modality, f, indent=4)


def _write_stats(stats: dict, meta_dir: Path) -> None:
    stats_path = meta_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


def _compute_stats(data: dict[str, list[np.ndarray]]) -> dict:
    stats = {}
    for key, values in data.items():
        if not values:
            continue
        stacked = np.vstack(values).astype(np.float32)
        stats[key] = {
            "mean": np.mean(stacked, axis=0).tolist(),
            "std": np.std(stacked, axis=0).tolist(),
            "min": np.min(stacked, axis=0).tolist(),
            "max": np.max(stacked, axis=0).tolist(),
            "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
            "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
        }
    return stats


def _write_info(
    meta_dir: Path,
    *,
    robot_type: str,
    motors: list[str],
    cameras: list[str],
    state_dim: int | None = None,
    action_dim: int | None = None,
    state_names: list[str] | None = None,
    action_names: list[str] | None = None,
    fps: float,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    chunk_size: int,
    video_info_by_camera: dict[str, dict],
) -> None:
    if state_dim is None:
        state_dim = len(motors)
    if action_dim is None:
        action_dim = len(motors)
    if state_names is not None and len(state_names) != state_dim:
        state_names = None
    if action_names is not None and len(action_names) != action_dim:
        action_names = None
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for cam in cameras:
        video_info = video_info_by_camera.get(cam, {})
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [
                video_info.get("video.height", 480),
                video_info.get("video.width", 640),
                3,
            ],
            "names": ["height", "width", "channels"],
            "info": video_info,
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": ceil(total_episodes / chunk_size) if total_episodes > 0 else 0,
        "total_videos": total_episodes * len(cameras),
    }

    info_path = meta_dir / "info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)


def _upload_v2_dataset(repo_id: str) -> None:
    dataset_path = HF_LEROBOT_HOME / repo_id
    if not dataset_path.exists():
        raise FileNotFoundError(f"Converted dataset not found at: {dataset_path}")

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_path),
        path_in_repo="",
        commit_message="Upload LeRobot v2.1 dataset",
    )


def json_to_lerobot_v2(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    modality_config_path: Path | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    fps: float = 25.0,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    force_conversion: bool = False,
) -> None:
    """Convert Unitree XR_teleoperate JSON data to LeRobot v2.1."""
    modality_override = None
    if modality_config_path is None:
        robot_config = _get_robot_config(robot_type)
    else:
        robot_config, modality_override = _load_modality_config(modality_config_path)

    if mode != "video":
        raise ValueError("Only mode='video' is supported for Gr00t 1.6.")

    dataset_root = _ensure_empty_dataset_root(repo_id)
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    json_dataset = JsonDataset(
        raw_dir,
        robot_config.json_state_data_name,
        robot_config.json_action_data_name,
        robot_config.camera_to_image_key,
    )
    motors = robot_config.motors
    cameras = robot_config.cameras

    chunk_size = 1000
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    total_frames = 0
    tasks: list[str] = []
    episode_tasks: list[str] = []
    episode_lengths: list[int] = []
    video_info_by_camera: dict[str, dict] = {}

    stats_buffers: dict[str, list[np.ndarray]] = {
        "action": [],
        "observation.state": [],
        "timestamp": [],
    }

    global_index = 0

    state_dim = None
    action_dim = None

    for i in tqdm.tqdm(range(len(json_dataset)), desc="Converting episodes"):
        episode = json_dataset.get_item(i)
        state = episode["state"].astype(np.float32)
        action = episode["action"].astype(np.float32)
        cameras_data = episode["cameras"]
        task = str(episode["task"]) if episode["task"] else "default"
        episode_length = episode["episode_length"]

        expected_cameras = cameras
        missing_cameras = [cam for cam in expected_cameras if cam not in cameras_data]
        if missing_cameras:
            raise ValueError(
                f"Episode {i} is missing cameras {missing_cameras}. "
                f"Available cameras: {list(cameras_data.keys())}. "
                "Check `camera_to_image_key` mappings and raw image files."
            )

        if state_dim is None:
            state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        elif state.shape[1] != state_dim:
            raise ValueError(
                f"State dimension mismatch in episode {i}: "
                f"expected {state_dim}, got {state.shape[1]}"
            )

        if action_dim is None:
            action_dim = action.shape[1] if len(action.shape) == 2 else action.shape[0]
        elif action.shape[1] != action_dim:
            raise ValueError(
                f"Action dimension mismatch in episode {i}: "
                f"expected {action_dim}, got {action.shape[1]}"
            )

        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)

        episode_tasks.append(task)
        episode_lengths.append(episode_length)
        total_frames += episode_length

        timestamps = (np.arange(episode_length, dtype=np.float32) / fps).astype(np.float32)
        frame_indices = np.arange(episode_length, dtype=np.int64)
        global_indices = np.arange(global_index, global_index + episode_length, dtype=np.int64)
        global_index += episode_length

        chunk_idx = i // chunk_size
        parquet_path = dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{i:06d}.parquet"

        _write_parquet(
            parquet_path,
            state=state,
            action=action,
            timestamps=timestamps,
            frame_indices=frame_indices,
            episode_index=i,
            global_indices=global_indices,
            task_index=task_index,
        )

        stats_buffers["action"].append(action)
        stats_buffers["observation.state"].append(state)
        stats_buffers["timestamp"].append(timestamps.reshape(-1, 1))

        for cam in cameras:
            video_dir = (
                dataset_root / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{cam}"
            )
            video_path = video_dir / f"episode_{i:06d}.mp4"
            video_info_by_camera[cam] = _write_video(cameras_data[cam], video_path, fps)

    _write_tasks(tasks, meta_dir)
    _write_episodes(episode_lengths, episode_tasks, meta_dir)
    _write_modality(motors, cameras, meta_dir, modality_override=modality_override)

    stats = _compute_stats(stats_buffers)
    _write_stats(stats, meta_dir)

    _write_info(
        meta_dir,
        robot_type=robot_type,
        motors=motors,
        cameras=cameras,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=motors if modality_override is None else list(modality_override["state"].keys()),
        action_names=motors if modality_override is None else list(modality_override["action"].keys()),
        fps=fps,
        total_episodes=len(json_dataset),
        total_frames=total_frames,
        total_tasks=len(tasks),
        chunk_size=chunk_size,
        video_info_by_camera=video_info_by_camera,
    )

    if push_to_hub:
        _upload_v2_dataset(repo_id)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot_v2)
