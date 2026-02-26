"""
G1 Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the Unitree G1 robot
using the GR00T Policy API.

Major responsibilities:
    - Initialize robot hardware from local robot_sdk controllers
    - Convert robot observations into GR00T VLA inputs
    - Query the GR00T policy server (PolicyClient)
    - Decode temporal model actions back into robot motor commands
    - Stream actions to the real robot in real time

This file is meant to be a simple, readable reference
for real-world policy debugging and demos.

Running example:
    uv run python gr00t/eval/real_robot/g1/eval_g1.py \
      --modality_config_path examples/g1_XRtele/modality_config.py \
      --modality_config_name unitree_g1_xrtele \
      --policy_host 127.0.0.1 \
      --policy_port 5555 \
      --action_horizon 8 \
      --control_hz 25 \
      --image_server_address 192.168.123.164 \
      --image_server_port 5556 \
      --lang_instruction "Pick up the object and place it in the tray."
"""

from dataclasses import asdict, dataclass
from importlib.util import module_from_spec, spec_from_file_location
import logging
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import shared_memory
from pprint import pformat
import select
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import draccus
import numpy as np
try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

from gr00t.policy.server_client import PolicyClient

from robot_sdk.make_robot import (
    process_images_and_observations,
    setup_image_client as setup_image_client_sdk,
    setup_robot_interface,
)
from robot_sdk.utils.utils import (
    _action_to_sized_vector,
    _to_numpy_image,
    recursive_add_extra_dim,
    to_list,
    to_scalar,
)
from image_server.image_client import ImageClient


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


def _read_current_waist_yaw(arm_ctrl: Any) -> float:
    return float(arm_ctrl.get_current_waist_yaw())


def setup_image_client_for_eval(cfg: "EvalConfig", camera_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Match eval_g1_lerobot setup style while keeping runtime image overrides
    in eval_g1 (without changing robot_sdk internals).
    """
    if getattr(cfg, "sim", False):
        return setup_image_client_sdk(cfg)

    tv_img_shape = (int(cfg.head_image_height), int(cfg.head_image_width), 3)
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    has_wrist_cam = bool(camera_keys) and any("wrist" in k for k in camera_keys)
    wrist_img_shape = None
    wrist_img_shm = None
    wrist_img_array = None

    if has_wrist_cam:
        wrist_img_shape = (
            int(cfg.wrist_image_height),
            int(cfg.wrist_image_width) * 2,
            3,
        )
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
            server_address=cfg.image_server_address,
            port=cfg.image_server_port,
        )
    else:
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            server_address=cfg.image_server_address,
            port=cfg.image_server_port,
        )
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    return {
        "tv_img_array": tv_img_array,
        "wrist_img_array": wrist_img_array,
        "tv_img_shape": tv_img_shape,
        "wrist_img_shape": wrist_img_shape,
        "is_binocular": False,
        "has_wrist_cam": has_wrist_cam,
        "image_thread": image_receive_thread,
        "shm_resources": [tv_img_shm, wrist_img_shm] if wrist_img_shm is not None else [tv_img_shm],
    }


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
            logging.info("Head camera frames detected in shared memory.")
        if wrist_baseline is not None and not got_wrist and not np.array_equal(
            wrist_baseline, wrist_img_array.reshape(-1)[::1000]
        ):
            got_wrist = True
            logging.info("Wrist camera frames detected in shared memory.")
        if got_head and got_wrist:
            return True
        time.sleep(0.05)

    thread_status = f"image thread alive={image_thread.is_alive()}" if image_thread else "image thread unavailable"
    logging.warning(
        "No camera frames detected after %.1fs; check image_server connection/resolution. (%s)",
        timeout_s,
        thread_status,
    )
    return False


def cleanup_image_resources(image_info: Optional[Dict[str, Any]]) -> None:
    """Close/unlink image shared memory segments created by this process."""
    if not image_info:
        return

    for shm in image_info.get("shm_resources", []):
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


class G1XRTeleAdapter:
    """
    Adapter between:
        - Raw robot observation dictionary
        - GR00T VLA input format
        - GR00T action chunk -> robot joint commands
    """

    def __init__(self, policy_client: PolicyClient, modality_configs: Dict[str, Any]):
        self.policy = policy_client
        self.camera_keys = modality_configs["video"].modality_keys
        self.robot_state_keys = modality_configs["state"].modality_keys
        self.action_keys = modality_configs["action"].modality_keys

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        model_obs: Dict[str, Any] = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Joint + hand state
        model_obs["state"] = {
            k: np.atleast_1d(np.asarray(obs[k], dtype=np.float32)) for k in self.robot_state_keys
        }

        # (3) Language
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict[str, np.ndarray], t: int) -> Dict[str, float]:
        if all(k in chunk for k in self.action_keys):
            out: Dict[str, float] = {}
            for k in self.action_keys:
                val = chunk[k][0][t]
                out[k] = float(val[0]) if isinstance(val, np.ndarray) and val.shape else float(val)
            return out

        any_key = next(iter(chunk.keys()))
        action_vec = chunk[any_key][0][t]
        return {joint_name: float(action_vec[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict[str, Any], horizon: Optional[int] = None) -> List[Dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, _ = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        total_horizon = action_chunk[any_key].shape[1]
        use_horizon = total_horizon if horizon is None else max(1, min(horizon, total_horizon))
        return [self.decode_action_chunk(action_chunk, t) for t in range(use_horizon)]


def _read_ee_side_states(ee_enabled: bool, ee_shared_mem: Dict[str, Any], ee_dof: int, ee_sides: tuple[str, ...]):
    side_states: Dict[str, np.ndarray] = {
        "left": np.zeros(0, dtype=np.float32),
        "right": np.zeros(0, dtype=np.float32),
    }
    if not ee_enabled or ee_dof <= 0 or not ee_shared_mem:
        return side_states

    with ee_shared_mem["lock"]:
        full_state = np.array(ee_shared_mem["state"][:], dtype=np.float32)

    offset = 0
    for side in ee_sides:
        side_states[side] = full_state[offset : offset + ee_dof]
        offset += ee_dof

    return side_states


def _build_policy_observation(
    camera_keys: List[str],
    observation: Dict[str, Any],
    current_arm_q: np.ndarray,
    current_waist_yaw: float,
    side_states: Dict[str, np.ndarray],
    lang_instruction: str,
) -> Dict[str, Any]:
    obs: Dict[str, Any] = {"lang": lang_instruction}

    for camera_key in camera_keys:
        image_key = f"observation.images.{camera_key}"
        obs[camera_key] = _to_numpy_image(observation.get(image_key))

    arm_q = np.asarray(current_arm_q, dtype=np.float32)
    for idx, key in enumerate(ARM_STATE_KEYS):
        obs[key] = float(arm_q[idx]) if idx < arm_q.shape[0] else 0.0

    left_state = np.asarray(side_states.get("left", np.zeros(0)), dtype=np.float32)
    right_state = np.asarray(side_states.get("right", np.zeros(0)), dtype=np.float32)
    for idx, key in enumerate(LEFT_HAND_STATE_KEYS):
        obs[key] = float(left_state[idx]) if idx < left_state.shape[0] else 0.0
    for idx, key in enumerate(RIGHT_HAND_STATE_KEYS):
        obs[key] = float(right_state[idx]) if idx < right_state.shape[0] else 0.0

    obs[WAIST_STATE_KEY] = float(current_waist_yaw)

    return obs


def _execute_action(
    action_dict: Dict[str, float],
    current_arm_q: np.ndarray,
    arm_ctrl: Any,
    arm_ik: Any,
    ee_enabled: bool,
    ee_dof: int,
    ee_sides: tuple[str, ...],
    ee_shared_mem: Dict[str, Any],
):
    arm_q_now = np.asarray(current_arm_q, dtype=np.float32)
    arm_action = np.array(
        [float(action_dict.get(key, arm_q_now[idx] if idx < arm_q_now.shape[0] else 0.0)) for idx, key in enumerate(ARM_STATE_KEYS)],
        dtype=np.float32,
    )
    tau = arm_ik.solve_tau(arm_action)
    arm_ctrl.ctrl_dual_arm(arm_action, tau)

    if WAIST_STATE_KEY in action_dict and hasattr(arm_ctrl, "ctrl_waist_yaw_abs"):
        arm_ctrl.ctrl_waist_yaw_abs(float(action_dict[WAIST_STATE_KEY]))

    if not ee_enabled or ee_dof <= 0:
        return

    left_action = np.array([float(action_dict.get(key, 0.0)) for key in LEFT_HAND_STATE_KEYS], dtype=np.float32)
    right_action = np.array([float(action_dict.get(key, 0.0)) for key in RIGHT_HAND_STATE_KEYS], dtype=np.float32)
    left_action = _action_to_sized_vector(left_action, ee_dof)
    right_action = _action_to_sized_vector(right_action, ee_dof)

    if isinstance(ee_shared_mem["left"], SynchronizedArray):
        if "left" in ee_sides:
            ee_shared_mem["left"][:] = to_list(left_action)
        else:
            ee_shared_mem["left"][:] = [0.0 for _ in ee_shared_mem["left"]]
        if "right" in ee_sides:
            ee_shared_mem["right"][:] = to_list(right_action)
        else:
            ee_shared_mem["right"][:] = [0.0 for _ in ee_shared_mem["right"]]
    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
        ee_shared_mem["left"].value = to_scalar(left_action) if "left" in ee_sides else 0.0
        ee_shared_mem["right"].value = to_scalar(right_action) if "right" in ee_sides else 0.0


@dataclass
class EvalConfig:
    modality_config_path: str = "examples/g1_XRtele/modality_config.py"
    modality_config_name: str = "unitree_g1_xrtele"
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Perform the task."
    control_hz: float = 25.0
    state_init_timeout_s: float = 5.0
    camera_init_timeout_s: float = 5.0
    require_start_keypress: bool = True
    image_server_address: str = "192.168.123.164"
    image_server_port: int = 5555
    head_image_height: int = 480
    head_image_width: int = 640
    wrist_image_height: int = 480
    wrist_image_width: int = 640

    # robot_sdk setup fields
    arm: str = "G1_29"
    ee: str = "dex3"
    ee_side: str = "both"
    sim: bool = False
    motion: bool = False

    # Optional DDS network interface name (e.g., "enx9c69d31ecd9b"); None uses autodetect.
    network_interface: Optional[str] = None
    enable_diagnostics: bool = False


@draccus.wrap()
def eval(cfg: EvalConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info(pformat(asdict(cfg)))

    # --- Setup policy wrapper ---
    spec = spec_from_file_location("modality_config", cfg.modality_config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load modality config from: {cfg.modality_config_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    modality_configs = getattr(module, cfg.modality_config_name)

    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = G1XRTeleAdapter(policy_client, modality_configs)

    image_info = None
    diag_file = None
    prev_arm_action: Optional[np.ndarray] = None
    pending_action_seq: List[Dict[str, float]] = []
    run_start_time = time.perf_counter()
    try:
        if cfg.enable_diagnostics:
            diag_path = f"eval_g1_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            diag_file = open(diag_path, "w")
            diag_file.write(
                "time_since_start_s,loop_elapsed_ms,sleep_time_ms,policy_infer_ms,action_delta_l2,tracking_error_l2,arm_dq_l2,missing_camera,using_hold_last\n"
            )
            diag_file.flush()
            logging.info("Diagnostics CSV: %s", diag_path)

        # --- Setup Phase ---
        image_info = setup_image_client_for_eval(cfg, camera_keys=policy.camera_keys)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl, arm_ik, ee_shared_mem, ee_dof, ee_sides = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "ee_dof", "ee_sides"]
        )

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
            timeout_s=cfg.camera_init_timeout_s,
        )

        if cfg.require_start_keypress:
            user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
            if user_input.strip().lower() != "s":
                logging.info("Start canceled by user input: %s", user_input)
                return

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        current_arm_q = arm_ctrl.get_current_dual_arm_q()
        tau = arm_ik.solve_tau(current_arm_q)
        arm_ctrl.ctrl_dual_arm(current_arm_q, tau)
        time.sleep(1.0)
        init_arm_q = np.asarray(current_arm_q, dtype=np.float32).copy()
        init_side_states = _read_ee_side_states(bool(cfg.ee), ee_shared_mem, ee_dof, ee_sides)

        # --- Run Main Loop ---
        logging.info("Starting evaluation loop at %.2f Hz.", cfg.control_hz)
        logging.info(
            "Runtime controls: SPACE (1) reset to initial pose and hold, SPACE (2) resume inference/motion. Press Ctrl+C to exit."
        )
        ee_enabled = bool(cfg.ee)
        reset_requested = False
        waiting_for_restart = False
        stdin_fd: Optional[int] = None
        old_term_attrs: Any = None
        hotkeys_enabled = False
        if sys.stdin.isatty() and termios is not None and tty is not None:
            try:
                stdin_fd = sys.stdin.fileno()
                old_term_attrs = termios.tcgetattr(stdin_fd)
                tty.setcbreak(stdin_fd)
                hotkeys_enabled = True
            except Exception:
                logging.exception("Failed to enable non-blocking keyboard input; continuing without hotkeys.")
        else:
            logging.warning("Keyboard hotkeys unavailable (stdin is not a POSIX TTY).")

        try:
            while True:
                if hotkeys_enabled:
                    while True:
                        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if not ready:
                            break
                        if sys.stdin.read(1) == " ":
                            pending_action_seq.clear()
                            prev_arm_action = None
                            if waiting_for_restart:
                                waiting_for_restart = False
                                #logging.info("Restart requested: resuming inference/motion.")
                            else:
                                reset_requested = True
                                waiting_for_restart = True
                                #logging.info("Reset requested: returning to initial pose. Press SPACE again to restart.")

                loop_start_time = time.perf_counter()
                missing_camera = False
                using_hold_last = False
                policy_infer_ms = 0.0
                action_dict: Optional[Dict[str, float]] = None

                if reset_requested:
                    tau = arm_ik.solve_tau(init_arm_q)
                    arm_ctrl.ctrl_dual_arm(init_arm_q, tau)
                    if ee_enabled and ee_dof > 0:
                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(init_side_states.get("left", np.zeros(ee_dof, dtype=np.float32)))
                            ee_shared_mem["right"][:] = to_list(init_side_states.get("right", np.zeros(ee_dof, dtype=np.float32)))
                        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["left"].value = to_scalar(
                                init_side_states.get("left", np.zeros(ee_dof, dtype=np.float32))
                            )
                            ee_shared_mem["right"].value = to_scalar(
                                init_side_states.get("right", np.zeros(ee_dof, dtype=np.float32))
                            )
                    time.sleep(1.0)
                    reset_requested = False

                if waiting_for_restart:
                    tau = arm_ik.solve_tau(init_arm_q)
                    arm_ctrl.ctrl_dual_arm(init_arm_q, tau)
                else:
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
                    side_states = _read_ee_side_states(ee_enabled, ee_shared_mem, ee_dof, ee_sides)
                    current_waist_yaw = _read_current_waist_yaw(arm_ctrl)
                    policy_obs = _build_policy_observation(
                        policy.camera_keys,
                        observation,
                        current_arm_q,
                        current_waist_yaw,
                        side_states,
                        cfg.lang_instruction,
                    )

                    if any(policy_obs.get(camera_key) is None for camera_key in policy.camera_keys):
                        logging.warning("Missing camera frame(s); skipping policy/action this cycle.")
                        missing_camera = True
                        pending_action_seq.clear()
                    else:
                        # 2. Execute cached chunk actions; refresh from policy when chunk is exhausted.
                        if pending_action_seq:
                            using_hold_last = True
                            action_dict = pending_action_seq.pop(0)
                        else:
                            t_policy = time.perf_counter()
                            action_seq = policy.get_action(policy_obs, horizon=max(1, cfg.action_horizon))
                            policy_infer_ms = (time.perf_counter() - t_policy) * 1000.0
                            if action_seq:
                                action_dict = action_seq[0]
                                pending_action_seq = action_seq[1:]
                            else:
                                logging.warning("Policy returned empty action sequence; skipping this cycle.")

                        if action_dict is not None:
                            # 3. Execute action through robot_sdk arm + EE interfaces
                            _execute_action(
                                action_dict=action_dict,
                                current_arm_q=current_arm_q,
                                arm_ctrl=arm_ctrl,
                                arm_ik=arm_ik,
                                ee_enabled=ee_enabled,
                                ee_dof=ee_dof,
                                ee_sides=ee_sides,
                                ee_shared_mem=ee_shared_mem,
                            )

                # Maintain frequency
                elapsed = time.perf_counter() - loop_start_time
                sleep_time = (1.0 / cfg.control_hz) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                arm_action_delta_l2 = 0.0
                arm_tracking_error_l2 = 0.0
                arm_dq_l2 = float(np.linalg.norm(np.asarray(arm_ctrl.get_current_dual_arm_dq(), dtype=np.float32)))
                if action_dict is not None:
                    arm_q_now = np.asarray(current_arm_q, dtype=np.float32)
                    arm_action = np.array(
                        [float(action_dict.get(key, arm_q_now[idx] if idx < arm_q_now.shape[0] else 0.0)) for idx, key in enumerate(ARM_STATE_KEYS)],
                        dtype=np.float32,
                    )
                    if prev_arm_action is not None:
                        arm_action_delta_l2 = float(np.linalg.norm(arm_action - prev_arm_action))
                    arm_tracking_error_l2 = float(np.linalg.norm(arm_action - arm_q_now))
                    prev_arm_action = arm_action

                if diag_file is not None:
                    diag_file.write(
                        f"{time.perf_counter() - run_start_time:.6f},{elapsed * 1000.0:.6f},{max(0.0, sleep_time) * 1000.0:.6f},"
                        f"{policy_infer_ms:.6f},{arm_action_delta_l2:.6f},{arm_tracking_error_l2:.6f},{arm_dq_l2:.6f},{int(missing_camera)},{int(using_hold_last)}\n"
                    )
                    diag_file.flush()
        finally:
            if hotkeys_enabled and stdin_fd is not None and old_term_attrs is not None:
                try:
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term_attrs)
                except Exception:
                    logging.exception("Failed to restore terminal settings.")

    except Exception:
        logging.exception("An error occurred during G1 evaluation.")
    finally:
        if diag_file is not None:
            diag_file.close()
        cleanup_image_resources(image_info)


if __name__ == "__main__":
    eval()
