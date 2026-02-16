import ctypes
import time
import logging
from pprint import pformat
from dataclasses import asdict
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from multiprocessing.sharedctypes import SynchronizedArray

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import (
    RerunLogger,
    visualization_data,
)

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def wait_for_camera_frames(tv_img_array, wrist_img_array=None, image_thread=None, timeout_s: float = 5.0) -> bool:
    """
    Light-weight check that the shared memory images are actually changing.
    Helps catch cases where the image client never connects or crashes early.
    """
    start = time.perf_counter()
    head_baseline = np.array(tv_img_array.reshape(-1)[::1000])
    wrist_baseline = np.array(wrist_img_array.reshape(-1)[::1000]) if wrist_img_array is not None else None
    got_head, got_wrist = False, wrist_baseline is None

    while time.perf_counter() - start < timeout_s:
        if not got_head and not np.array_equal(head_baseline, tv_img_array.reshape(-1)[::1000]):
            got_head = True
            logger_mp.info("Head camera frames detected in shared memory.")
        if wrist_baseline is not None and not got_wrist and not np.array_equal(
            wrist_baseline, wrist_img_array.reshape(-1)[::1000]
        ):
            got_wrist = True
            logger_mp.info("Wrist camera frames detected in shared memory.")
        if got_head and got_wrist:
            return True
        time.sleep(0.05)

    thread_status = f"image thread alive={image_thread.is_alive()}" if image_thread else "image thread unavailable"
    logger_mp.warning(
        "No camera frames detected after %.1fs; check image_server connection/resolution. (%s)",
        timeout_s,
        thread_status,
    )
    return False


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | nn.Module | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()
    else:
        rerun_logger = None

    # Reset policy and processors if they are provided
    if policy is not None:
        policy.reset()
    if preprocessor is not None:
        preprocessor.reset()
    if postprocessor is not None:
        postprocessor.reset()

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof, ee_sides = (
            robot_interface[key]
            for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof", "ee_sides"]
        )

        canonical_sides = ("left", "right")
        policy_total_ee_dof = ee_dof * len(canonical_sides) if cfg.ee else 0

        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # Make sure we are actually getting frames
        wait_for_camera_frames(
            tv_img_array,
            wrist_img_array if has_wrist_cam else None,
            image_info.get("image_thread"),
        )

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        full_state = None

        ee_side = getattr(cfg, "ee_side", None)

        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = arm_ik.solve_tau(init_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move

            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()

                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array,
                    wrist_img_array,
                    tv_img_shape,
                    wrist_img_shape,
                    is_binocular,
                    has_wrist_cam,
                    arm_ctrl,
                )

                side_states: dict[str, np.ndarray] = {}
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                    offset = 0
                    for side in ee_sides:
                        side_states[side] = full_state[offset : offset + ee_dof]
                        offset += ee_dof
                    # Fill missing side(s) with zeros to keep policy input dimension consistent
                    for side in canonical_sides:
                        side_states.setdefault(side, np.zeros(ee_dof))

                state_vec_parts = [current_arm_q]
                if cfg.ee:
                    state_vec_parts += [side_states[s] for s in canonical_sides]

                state_tensor = torch.from_numpy(np.concatenate(state_vec_parts, axis=0)).float()

                # Default: full state (e.g. 28 DoF)
                # Optional: convert to 14-DoF if we only control left side
                if cfg.ee and ee_side == "left":
                    # state_tensor shape: [28]
                    # 0:7   -> left arm
                    # 7:14  -> right arm
                    # 14:21 -> left hand
                    # 21:28 -> right hand
                    left_arm = state_tensor[:7]
                    left_hand = state_tensor[14:21]
                    state_tensor14 = torch.cat([left_arm, left_hand], dim=0)  # [14]
                    observation["observation.state"] = state_tensor14
                else:
                    observation["observation.state"] = state_tensor

                # 2. Get Action from Policy (with pre/post-processing)
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=getattr(cfg, "robot_type", None),
                )

                # Optional: map 14-DoF action back into 28-DoF command vector
                if cfg.ee and ee_side == "left":
                    action28 = state_tensor.clone()
                    # Map predicted 14 DoF back into the left arm+hand slots
                    action28[:7] = action[:7]        # left arm
                    action28[14:21] = action[7:14]   # left hand
                    action_np = action28.cpu().numpy()
                else:
                    action_np = action.cpu().numpy()

                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    ee_action_vec = action_np[ee_action_start_idx : ee_action_start_idx + policy_total_ee_dof]

                    side_actions: dict[str, np.ndarray] = {}
                    offset = 0
                    for side in canonical_sides:
                        side_actions[side] = ee_action_vec[offset : offset + ee_dof]
                        offset += ee_dof

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        if "left" in ee_sides:
                            ee_shared_mem["left"][:] = to_list(side_actions["left"])
                        else:
                            ee_shared_mem["left"][:] = [0.0 for _ in ee_shared_mem["left"]]
                        if "right" in ee_sides:
                            ee_shared_mem["right"][:] = to_list(side_actions["right"])
                        else:
                            ee_shared_mem["right"][:] = [0.0 for _ in ee_shared_mem["right"]]
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(side_actions["left"]) if "left" in ee_sides else 0.0
                        ee_shared_mem["right"].value = (
                            to_scalar(side_actions["right"]) if "right" in ee_sides else 0.0
                        )

                if cfg.visualization and rerun_logger is not None:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)

                idx += 1

                # Maintain frequency
                elapsed = time.perf_counter() - loop_start_time
                sleep_time = (1.0 / cfg.frequency) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    # Some configs (like EvalRealConfig) may not have rename_map; default to empty dict
    rename_map = getattr(cfg, "rename_map", {})

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(
            cfg=cfg,
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
