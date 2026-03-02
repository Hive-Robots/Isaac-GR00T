from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.data.types import ActionConfig
from gr00t.data.types import ActionFormat
from gr00t.data.types import ActionRepresentation
from gr00t.data.types import ActionType

# Modality configuration for the Unitree G1 dataset (xr_tele) version.
unitree_g1_xrtele_yaw_hand_IO = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
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
            "left_trig",
            "right_trig",
            "kWaistYaw",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
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
            "left_trig",
            "right_trig",
            "kWaistYaw",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in range(17)
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(unitree_g1_xrtele_yaw_hand_IO, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
