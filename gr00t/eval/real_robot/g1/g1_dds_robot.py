from threading import Lock
import time
from typing import Any, Dict, List, Optional

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


G1_ARM_KEY_TO_INDEX = {
    "kLeftShoulderPitch": 15,
    "kLeftShoulderRoll": 16,
    "kLeftShoulderYaw": 17,
    "kLeftElbow": 18,
    "kLeftWristRoll": 19,
    "kLeftWristPitch": 20,
    "kLeftWristYaw": 21,
    "kRightShoulderPitch": 22,
    "kRightShoulderRoll": 23,
    "kRightShoulderYaw": 24,
    "kRightElbow": 25,
    "kRightWristRoll": 26,
    "kRightWristPitch": 27,
    "kRightWristYaw": 28,
}

LEFT_HAND_KEYS = [
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
]

RIGHT_HAND_KEYS = [
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]


def make_hand_mode(motor_index: int) -> int:
    status = 0x01
    timeout = 0x01
    mode = motor_index & 0x0F
    mode |= status << 4
    mode |= timeout << 7
    return mode


class G1DDSRobot:
    def __init__(self, network_interface: Optional[str], camera_keys: List[str]):
        self.network_interface = network_interface
        self.camera_keys = camera_keys

        self._lock = Lock()
        self._low_state = None
        self._left_hand_state = None
        self._right_hand_state = None

        self._crc = CRC()
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()

        self._arm_kp = 60.0
        self._arm_kd = 1.5
        self._hand_kp = [2.0] + [0.5] * 6
        self._hand_kd = [0.1] * 7

        self._arm_pub = None
        self._left_hand_pub = None
        self._right_hand_pub = None
        self._low_sub = None
        self._left_hand_sub = None
        self._right_hand_sub = None

    def connect(self, state_init_timeout_s: float = 5.0):
        if self.network_interface:
            ChannelFactoryInitialize(0, self.network_interface)
        else:
            ChannelFactoryInitialize(0)

        self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._left_hand_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self._right_hand_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)

        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._left_hand_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self._right_hand_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)

        self._arm_pub.Init()
        self._left_hand_pub.Init()
        self._right_hand_pub.Init()
        self._low_sub.Init(self._on_low_state, 10)
        self._left_hand_sub.Init(self._on_left_hand_state, 10)
        self._right_hand_sub.Init(self._on_right_hand_state, 10)

        deadline = time.time() + state_init_timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._low_state is not None:
                    return
            time.sleep(0.01)

        raise RuntimeError("Timed out waiting for first rt/lowstate message from G1")

    def _on_low_state(self, msg: LowState_):
        with self._lock:
            self._low_state = msg

    def _on_left_hand_state(self, msg: HandState_):
        with self._lock:
            self._left_hand_state = msg

    def _on_right_hand_state(self, msg: HandState_):
        with self._lock:
            self._right_hand_state = msg

    def get_observation(self) -> Dict[str, Any]:
        with self._lock:
            low_state = self._low_state
            left_hand = self._left_hand_state
            right_hand = self._right_hand_state

        if low_state is None:
            raise RuntimeError("No low_state received yet")

        obs = {k: np.zeros((480, 640, 3), dtype=np.uint8) for k in self.camera_keys}

        for key, idx in G1_ARM_KEY_TO_INDEX.items():
            obs[key] = float(low_state.motor_state[idx].q)

        for i, key in enumerate(LEFT_HAND_KEYS):
            if left_hand is not None and i < len(left_hand.motor_state):
                obs[key] = float(left_hand.motor_state[i].q)
            else:
                obs[key] = 0.0

        for i, key in enumerate(RIGHT_HAND_KEYS):
            if right_hand is not None and i < len(right_hand.motor_state):
                obs[key] = float(right_hand.motor_state[i].q)
            else:
                obs[key] = 0.0

        return obs

    def send_action(self, action_dict: Dict[str, float]):
        with self._lock:
            low_state = self._low_state
            left_hand_state = self._left_hand_state
            right_hand_state = self._right_hand_state

        if low_state is None:
            return

        self._low_cmd.mode_machine = low_state.mode_machine
        self._low_cmd.motor_cmd[29].q = 1.0

        for key, idx in G1_ARM_KEY_TO_INDEX.items():
            if key not in action_dict:
                continue
            motor = self._low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.q = float(action_dict[key])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self._arm_kp
            motor.kd = self._arm_kd

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._arm_pub.Write(self._low_cmd)

        left_cmd = unitree_hg_msg_dds__HandCmd_()
        right_cmd = unitree_hg_msg_dds__HandCmd_()

        for i in range(7):
            left_motor = left_cmd.motor_cmd[i]
            right_motor = right_cmd.motor_cmd[i]

            left_motor.mode = make_hand_mode(i)
            right_motor.mode = make_hand_mode(i)

            left_default = 0.0
            right_default = 0.0
            if left_hand_state is not None and i < len(left_hand_state.motor_state):
                left_default = float(left_hand_state.motor_state[i].q)
            if right_hand_state is not None and i < len(right_hand_state.motor_state):
                right_default = float(right_hand_state.motor_state[i].q)

            left_motor.q = float(action_dict.get(LEFT_HAND_KEYS[i], left_default))
            right_motor.q = float(action_dict.get(RIGHT_HAND_KEYS[i], right_default))

            left_motor.dq = 0.0
            right_motor.dq = 0.0
            left_motor.tau = 0.0
            right_motor.tau = 0.0
            left_motor.kp = self._hand_kp[i]
            right_motor.kp = self._hand_kp[i]
            left_motor.kd = self._hand_kd[i]
            right_motor.kd = self._hand_kd[i]

        self._left_hand_pub.Write(left_cmd)
        self._right_hand_pub.Write(right_cmd)
