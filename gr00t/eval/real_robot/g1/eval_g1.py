"""
G1 Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the Unitree G1 robot
using the GR00T Policy API.

Major responsibilities:
    • Initialize robot hardware
    • Convert robot observations into GR00T VLA inputs
    • Query the GR00T policy server (PolicyClient)
    • Decode multi-step (temporal) model actions back into robot motor commands
    • Stream actions to the real robot in real time

This file is meant to be a simple, readable reference
for real-world policy debugging and demos.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import time
from typing import Any, Dict, List

import draccus
import numpy as np

from importlib.util import module_from_spec, spec_from_file_location
from gr00t.policy.server_client import PolicyClient

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


class G1XRTeleAdapter:
    """
    Adapter between:
        • Raw robot observation dictionary
        • GR00T VLA input format
        • GR00T action chunk → robot joint commands

    Responsible for:
        • Packaging camera frames as obs["video"]
        • Building obs["state"] for G1 joints/hands
        • Adding language instruction
        • Adding batch/time dimensions
        • Decoding model action chunks into real robot actions
    """

    def __init__(self, policy_client: PolicyClient, modality_configs: Dict[str, Any]):
        self.policy = policy_client

        # XR teleop ordering used for BOTH training + robot execution
        self.camera_keys = modality_configs["video"].modality_keys
        self.robot_state_keys = modality_configs["state"].modality_keys
        self.action_keys = modality_configs["action"].modality_keys

    # -------------------------------------------------------------------------
    # Observation → Model Input
    # -------------------------------------------------------------------------
    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """
        Convert raw robot observation dict into the structured GR00T VLA input.
        """
        model_obs: Dict[str, Any] = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Joint + hand state
        model_obs["state"] = {k: np.asarray(obs[k], dtype=np.float32) for k in self.robot_state_keys}

        # (3) Language
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk → Robot Motor Commands
    # -------------------------------------------------------------------------
    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """
        chunk["action"]: (B, T, D)

        Convert to:
            {
                "kLeftShoulderPitch": val,
                ...
            }
        for timestep t.
        """
        if all(k in chunk for k in self.action_keys):
            out: Dict[str, float] = {}
            for k in self.action_keys:
                val = chunk[k][0][t]
                out[k] = float(val[0]) if isinstance(val, np.ndarray) and val.shape else float(val)
            return out

        # Fallback: single vector output
        any_key = next(iter(chunk.keys()))
        action_vec = chunk[any_key][0][t]  # (D,)
        return {joint_name: float(action_vec[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        """
        Returns a list of robot motor commands (one per model timestep).
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Evaluation Config
# =============================================================================


@dataclass
class EvalConfig:
    """
    Command-line configuration for real-robot policy evaluation.
    """

    modality_config_path: str = "examples/g1_XRtele/modality_config.py"
    modality_config_name: str = "unitree_g1_xrtele"
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Perform the task."
    control_hz: float = 30.0


# =============================================================================
# Main Eval Loop
# =============================================================================


@draccus.wrap()
def eval(cfg: EvalConfig):
    """
    Main entry point for real-robot policy evaluation.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(pformat(asdict(cfg)))

    # -------------------------------------------------------------------------
    # 1. Initialize Robot Hardware
    # -------------------------------------------------------------------------
    # TODO: Initialize Unitree G1 hardware and SDK here.
    # robot = ...
    # robot.connect()

    # -------------------------------------------------------------------------
    # 2. Initialize Policy Wrapper + Client
    # -------------------------------------------------------------------------
    spec = spec_from_file_location("modality_config", cfg.modality_config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load modality config from: {cfg.modality_config_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    modality_configs = getattr(module, cfg.modality_config_name)

    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = G1XRTeleAdapter(policy_client, modality_configs)

    logging.info('Policy ready with instruction: "%s"', cfg.lang_instruction)

    # -------------------------------------------------------------------------
    # 3. Main real-time control loop
    # -------------------------------------------------------------------------
    while True:
        # TODO: Collect observation from Unitree G1.
        # obs = robot.get_observation()
        obs = {k: np.zeros((480, 640, 3), dtype=np.uint8) for k in policy.camera_keys}
        for k in policy.robot_state_keys:
            obs[k] = 0.0
        obs["lang"] = cfg.lang_instruction  # insert language

        # obs = {
        #     "cam_left_high": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "kLeftShoulderPitch": 0.0,
        #     "kLeftShoulderRoll": 0.0,
        #     "kLeftShoulderYaw": 0.0,
        #     "kLeftElbow": 0.0,
        #     "kLeftWristRoll": 0.0,
        #     "kLeftWristPitch": 0.0,
        #     "kLeftWristYaw": 0.0,
        #     "kRightShoulderPitch": 0.0,
        #     "kRightShoulderRoll": 0.0,
        #     "kRightShoulderYaw": 0.0,
        #     "kRightElbow": 0.0,
        #     "kRightWristRoll": 0.0,
        #     "kRightWristPitch": 0.0,
        #     "kRightWristYaw": 0.0,
        #     "kLeftHandThumb0": 0.0,
        #     "kLeftHandThumb1": 0.0,
        #     "kLeftHandThumb2": 0.0,
        #     "kLeftHandMiddle0": 0.0,
        #     "kLeftHandMiddle1": 0.0,
        #     "kLeftHandIndex0": 0.0,
        #     "kLeftHandIndex1": 0.0,
        #     "kRightHandThumb0": 0.0,
        #     "kRightHandThumb1": 0.0,
        #     "kRightHandThumb2": 0.0,
        #     "kRightHandIndex0": 0.0,
        #     "kRightHandIndex1": 0.0,
        #     "kRightHandMiddle0": 0.0,
        #     "kRightHandMiddle1": 0.0,
        #     "lang": cfg.lang_instruction,
        # }

        actions = policy.get_action(obs)

        for i, action_dict in enumerate(actions[: cfg.action_horizon]):
            tic = time.time()
            logging.info("action[%d]: %s", i, action_dict)
            # action_dict = {
            #     "kLeftShoulderPitch": 0.0,
            #     "kLeftShoulderRoll": 0.0,
            #     "kLeftShoulderYaw": 0.0,
            #     "kLeftElbow": 0.0,
            #     "kLeftWristRoll": 0.0,
            #     "kLeftWristPitch": 0.0,
            #     "kLeftWristYaw": 0.0,
            #     "kRightShoulderPitch": 0.0,
            #     "kRightShoulderRoll": 0.0,
            #     "kRightShoulderYaw": 0.0,
            #     "kRightElbow": 0.0,
            #     "kRightWristRoll": 0.0,
            #     "kRightWristPitch": 0.0,
            #     "kRightWristYaw": 0.0,
            #     "kLeftHandThumb0": 0.0,
            #     "kLeftHandThumb1": 0.0,
            #     "kLeftHandThumb2": 0.0,
            #     "kLeftHandMiddle0": 0.0,
            #     "kLeftHandMiddle1": 0.0,
            #     "kLeftHandIndex0": 0.0,
            #     "kLeftHandIndex1": 0.0,
            #     "kRightHandThumb0": 0.0,
            #     "kRightHandThumb1": 0.0,
            #     "kRightHandThumb2": 0.0,
            #     "kRightHandIndex0": 0.0,
            #     "kRightHandIndex1": 0.0,
            #     "kRightHandMiddle0": 0.0,
            #     "kRightHandMiddle1": 0.0,
            # }
            # TODO: Send action_dict to Unitree G1.
            # robot.send_action(action_dict)
            toc = time.time()
            if toc - tic < 1.0 / cfg.control_hz:
                time.sleep(1.0 / cfg.control_hz - (toc - tic))


if __name__ == "__main__":
    eval()
