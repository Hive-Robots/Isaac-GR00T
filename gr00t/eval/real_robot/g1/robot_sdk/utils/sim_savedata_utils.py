# for simulation
import torch
import numpy as np
import logging_mp
from robot_sdk.utils.utils import (
    reset_policy,
)
from robot_sdk.make_robot import (
    publish_reset_category,
)
from dataclasses import dataclass
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
import time

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def process_data_add(episode_writer, observation_image, current_arm_q, ee_state, action, arm_dof, ee_dof, ee_sides):
    if episode_writer is None:
        return
    if (
        observation_image is not None
        and current_arm_q is not None
        and ee_state is not None
        and action is not None
        and arm_dof is not None
        and ee_dof is not None
    ):
        # Convert tensors to numpy arrays for JSON serialization
        if torch.is_tensor(current_arm_q):
            current_arm_q = current_arm_q.detach().cpu().numpy()
        if torch.is_tensor(ee_state):
            ee_state = ee_state.detach().cpu().numpy()
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        colors = {}
        i = 0
        for key, value in observation_image.items():
            if "images" in key:
                if value is not None:
                    # Convert PyTorch tensor to numpy array for OpenCV compatibility
                    if torch.is_tensor(value):
                        # Convert tensor to numpy array and ensure correct format for OpenCV
                        img_array = value.detach().cpu().numpy()
                        # If the image is in CHW format (channels first), convert to HWC format (channels last)
                        if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                            img_array = np.transpose(img_array, (1, 2, 0))
                        # Ensure the array is in uint8 format for OpenCV
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:  # Normalized values [0, 1]
                                img_array = (img_array * 255).astype(np.uint8)
                            else:  # Values already in [0, 255] range
                                img_array = img_array.astype(np.uint8)
                                # Keep original RGB format - no color channel conversion needed
                        colors[f"color_{i}"] = img_array
                    else:
                        colors[f"color_{i}"] = value
                    i += 1
        canonical_sides = ("left", "right")
        active_sides = tuple(ee_sides) if ee_sides else canonical_sides
        ee_state_parts = {"left": [], "right": []}
        offset = 0
        for side in active_sides:
            ee_state_parts[side] = ee_state[offset : offset + ee_dof].tolist()
            offset += ee_dof
        for side in canonical_sides:
            if not ee_state_parts[side]:
                ee_state_parts[side] = [0.0] * ee_dof

        states = {
            "left_arm": {
                "qpos": current_arm_q[: arm_dof // 2].tolist(),  # numpy.array -> list
                "qvel": [],
                "torque": [],
            },
            "right_arm": {
                "qpos": current_arm_q[arm_dof // 2 :].tolist(),
                "qvel": [],
                "torque": [],
            },
            "left_ee": {
                "qpos": ee_state[:ee_dof].tolist(),
                "qvel": [],
                "torque": [],
            },
            "right_ee": {
                "qpos": ee_state[ee_dof:].tolist(),
                "qvel": [],
                "torque": [],
            },
            "body": {
                "qpos": [],
            },
        }
        actions = {
            "left_arm": {
                "qpos": action[: arm_dof // 2].tolist(),
                "qvel": [],
                "torque": [],
            },
            "right_arm": {
                "qpos": action[arm_dof // 2 :].tolist(),
                "qvel": [],
                "torque": [],
            },
            "left_ee": {"qpos": [], "qvel": [], "torque": []},
            "right_ee": {"qpos": [], "qvel": [], "torque": []},
            "body": {
                "qpos": [],
            },
        }
        offset = 0
        for side in canonical_sides:
            actions[f"{side}_ee"]["qpos"] = action[arm_dof + offset : arm_dof + offset + ee_dof].tolist()
            offset += ee_dof

        states["left_ee"]["qpos"] = ee_state_parts["left"]
        states["right_ee"]["qpos"] = ee_state_parts["right"]
        episode_writer.add_item(colors, states=states, actions=actions)


def process_data_save(episode_writer, result):
    """Processes data and saves it."""
    if episode_writer is None:
        return
    episode_writer.save_episode(result)


def is_success(
    sim_reward_subscriber,
    episode_writer,
    reset_pose_publisher,
    policy,
    cfg,
    reward_stats,
    init_arm_pose,
    robot_interface,
):
    # logger_mp.info(f"arm_action {arm_action}, tau {tau}")
    logger_mp.info(
            f"Iteration {reward_stats['episode_num']} currently with reward {reward_stats['reward_sum']}"
        )
    if sim_reward_subscriber:
        data = sim_reward_subscriber.read_data()
        if data is not None:
            if int(data["rewards"][0]) == 1:
                reward_stats["reward_sum"] += 1
        sim_reward_subscriber.reset_data()
    # success
    if reward_stats["reward_sum"] >= 25: #HIVE-INFO: analyzing reward calculation, this will never happen
        process_data_save(episode_writer, "success")
        logger_mp.info(
            f"Iteration {reward_stats['episode_num']} finished with reward {reward_stats['reward_sum']},save data..."
        )
        reward_stats["episode_num"] = -1
        reward_stats["reward_sum"] = 0
        time.sleep(1)
        publish_reset_category(1, reset_pose_publisher)
        time.sleep(1)
        reset_policy(policy)
        sim_reward_subscriber.reset_data()
    # fail
    elif reward_stats["episode_num"] > cfg.max_episodes: #HIVE-INFO: If step counter exceeds exceeds cfg.episode_num, episode is a failure. Here is where robot is restarted
        process_data_save(episode_writer, "fail")
        logger_mp.info(f"Iteration {reward_stats['episode_num']} finished with reward {reward_stats['reward_sum']}")
        reward_stats["episode_num"] = -1
        reward_stats["reward_sum"] = 0
        reset_policy(policy)
        sim_reward_subscriber.reset_data()
        logger_mp.info("Initializing robot to starting pose...")
        tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
        robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(1)
        publish_reset_category(1, reset_pose_publisher)
        time.sleep(1)
        reset_policy(policy)
        sim_reward_subscriber.reset_data()
        time.sleep(1)


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex3"  # dex3, dex1, inspire1, brainco
    ee_side: str = "both"  # both, left, right

    # Mode flags
    motion: bool = False
    headless: bool = False
    sim: bool = True
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False
    save_data: bool = False
    task_dir: str = "./data"
    max_episodes: int = 1200 #HIVE-INFO: max episode number before reset inference

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logger_mp.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
