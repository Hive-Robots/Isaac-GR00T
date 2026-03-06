from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from robot_sdk.utils.episode_writer import EpisodeWriter

ARM_STATE_KEYS = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
]

LEFT_HAND_STATE_KEYS = [
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
]

RIGHT_HAND_STATE_KEYS = [
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]

WAIST_STATE_KEY = "kWaistYaw"
LEFT_TRIG_KEY = "left_trig"
RIGHT_TRIG_KEY = "right_trig"
BASE_VEL_KEYS = ["vx", "vy", "vyaw"]


def to_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _vector_from_action(action_dict: Dict[str, float], keys: list[str], fallback: np.ndarray) -> list[float]:
    fb = np.asarray(fallback, dtype=np.float32)
    out: list[float] = []
    for idx, key in enumerate(keys):
        default = float(fb[idx]) if idx < fb.shape[0] else 0.0
        out.append(float(action_dict.get(key, default)))
    return out


def _base_qvel_from_action(action_dict: Dict[str, float]) -> list[float]:
    return [float(action_dict.get(key, 0.0)) for key in BASE_VEL_KEYS]


def _binary_trigger(value: float) -> int:
    return int(np.clip(np.rint(float(value)), 0.0, 1.0))


def _trigger_from_obs_or_action(policy_obs: Dict[str, Any], action_dict: Dict[str, float], key: str) -> int:
    if key in policy_obs:
        return _binary_trigger(policy_obs[key])
    return _binary_trigger(action_dict.get(key, 0.0))


def _torque_from_side(torques: Optional[Dict[str, np.ndarray]], side: str) -> list[float]:
    if not torques:
        return []
    arr = np.asarray(torques.get(side, np.zeros(0, dtype=np.float32)), dtype=np.float32)
    return to_float_list(arr)


@dataclass
class G1EpisodeRecorder:
    task_dir: str
    task_goal: str
    control_hz: float
    image_width: int
    image_height: int
    rerun_log: bool = False

    def __post_init__(self) -> None:
        self.writer = EpisodeWriter(
            task_dir=self.task_dir,
            task_goal=self.task_goal,
            frequency=float(self.control_hz),
            image_size=[int(self.image_width), int(self.image_height)],
            rerun_log=bool(self.rerun_log),
        )
        self.writer.info.setdefault("joint_names", {})
        self.writer.info["joint_names"]["left_trig"] = [LEFT_TRIG_KEY]
        self.writer.info["joint_names"]["right_trig"] = [RIGHT_TRIG_KEY]
        task_name = Path(self.task_dir).name.strip() or "eval_g1_loop"
        self.writer.text["task_name"] = task_name
        self.writer.text["prompt_idx"] = 0
        self.writer.text["task_idx"] = 0
        self.is_recording = False

    def start_episode(self) -> None:
        if self.is_recording:
            return
        if self.writer.create_episode():
            self.is_recording = True

    def stop_episode(self) -> None:
        if not self.is_recording:
            return
        self.writer.save_episode()
        self.is_recording = False

    def close(self) -> None:
        if self.is_recording:
            self.writer.save_episode()
            self.is_recording = False
        self.writer.close()

    def record_step(
        self,
        policy_obs: Dict[str, Any],
        camera_keys: list[str],
        current_arm_q: np.ndarray,
        current_waist_yaw: float,
        side_states: Dict[str, np.ndarray],
        action_dict: Dict[str, float],
        torques: Optional[Dict[str, np.ndarray]] = None,
        tactiles: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if not self.is_recording:
            return

        colors: dict[str, np.ndarray] = {}
        color_idx = 0
        for camera_key in camera_keys:
            image = policy_obs.get(camera_key)
            if image is None:
                continue
            colors[f"color_{color_idx}"] = np.asarray(image)
            color_idx += 1

        arm_q = np.asarray(current_arm_q, dtype=np.float32)
        left_state = np.asarray(side_states.get("left", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        right_state = np.asarray(side_states.get("right", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        left_torque = _torque_from_side(torques, "left")
        right_torque = _torque_from_side(torques, "right")
        left_trig_state = _trigger_from_obs_or_action(policy_obs, action_dict, LEFT_TRIG_KEY)
        right_trig_state = _trigger_from_obs_or_action(policy_obs, action_dict, RIGHT_TRIG_KEY)

        states = {
            "left_arm": {"qpos": to_float_list(arm_q[:7]), "qvel": [], "torque": []},
            "right_arm": {"qpos": to_float_list(arm_q[7:14]), "qvel": [], "torque": []},
            "left_ee": {"qpos": to_float_list(left_state), "qvel": [], "torque": left_torque},
            "right_ee": {"qpos": to_float_list(right_state), "qvel": [], "torque": right_torque},
            "waist": {"qpos": [float(current_waist_yaw)], "qvel": []},
            "base": {"qpos": [], "qvel": _base_qvel_from_action(action_dict)},
            "left_trig": {"qpos": [left_trig_state], "qvel": [], "torque": []},
            "right_trig": {"qpos": [right_trig_state], "qvel": [], "torque": []},
        }

        actions = {
            "left_arm": {
                "qpos": _vector_from_action(action_dict, ARM_STATE_KEYS[:7], arm_q[:7]),
                "qvel": [],
                "torque": [],
            },
            "right_arm": {
                "qpos": _vector_from_action(action_dict, ARM_STATE_KEYS[7:], arm_q[7:14]),
                "qvel": [],
                "torque": [],
            },
            "left_ee": {
                "qpos": _vector_from_action(action_dict, LEFT_HAND_STATE_KEYS, left_state),
                "qvel": [],
                "torque": left_torque,
            },
            "right_ee": {
                "qpos": _vector_from_action(action_dict, RIGHT_HAND_STATE_KEYS, right_state),
                "qvel": [],
                "torque": right_torque,
            },
            "waist": {
                "qpos": [float(action_dict.get(WAIST_STATE_KEY, current_waist_yaw))],
                "qvel": [],
            },
            "base": {"qpos": [], "qvel": _base_qvel_from_action(action_dict)},
            "left_trig": {
                "qpos": [_binary_trigger(action_dict.get(LEFT_TRIG_KEY, 0.0))]
            },
            "right_trig": {
                "qpos": [_binary_trigger(action_dict.get(RIGHT_TRIG_KEY, 0.0))]
            },
        }

        frame_torques: Optional[dict[str, list[float]]] = None
        if left_torque or right_torque:
            frame_torques = {
                "left_ee": left_torque,
                "right_ee": right_torque,
            }

        frame_tactiles: Optional[dict[str, list[float]]] = None
        if tactiles:
            left_tactile = to_float_list(np.asarray(tactiles.get("left", np.zeros(0, dtype=np.float32)), dtype=np.float32))
            right_tactile = to_float_list(np.asarray(tactiles.get("right", np.zeros(0, dtype=np.float32)), dtype=np.float32))
            if left_tactile or right_tactile:
                frame_tactiles = {
                    "left_ee": left_tactile,
                    "right_ee": right_tactile,
                }

        self.writer.add_item(
            colors=colors,
            depths={},
            states=states,
            actions=actions,
            tactiles=frame_tactiles,
            torques=frame_torques,
            audios=None,
            sim_state=None,
        )
