# G1 Policy Client Container

Minimal Docker image for running the inference client directly on the robot:
- `gr00t/eval/real_robot/g1/eval_g1.py`
- `gr00t/eval/real_robot/g1/eval_g1_loop.py`

This image is intended to be built directly on the Unitree G1 Jetson Orin robot (ARM64).

## Build

From this directory:

```bash
bash build.sh
```

Optional image name:

```bash
IMAGE_NAME=g1-policy-client:jetson bash build.sh
```

## Run

Interactive shell:

```bash
bash run.sh
```

## Recommended Topology (Your Setup)

- `eval_g1` / `eval_g1_loop`: runs on the robot inside this container
- `run_gr00t_server.py`: runs on your computer (same Ethernet network)

### Server side (your computer)

Use:
- `--host 0.0.0.0`
- `--port 5555`

```bash
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/g1_XRtele/modality_config.py\
  --model-path /mnt/sata1/gr00t16/... \
  --host 0.0.0.0 \
  --port 5555
```

### Eval side (robot container)

Use:
- `--network_interface`: robot NIC carrying `192.168.123.164` (often `eth0`)
- `--image_server_address 127.0.0.1` (or `192.168.123.164`)
- `--image_server_port 5555` (matches `image_server.py` default)
- `--policy_host <YOUR_COMPUTER_ETHERNET_IP>`
- `--policy_port 5555`

Run loop eval directly:

```bash
bash run.sh python gr00t/eval/real_robot/g1/eval_g1_loop.py \
  --modality_config_path examples/g1_XRtele/modality_config.py \
  --modality_config_name unitree_g1_xrtele \
  --policy_host 192.168.123.222 \
  --policy_port 5555 \
  --action_horizon 16 \
  --control_hz 25 \
  --network_interface eth0 \
  --image_server_address 127.0.0.1 \
  --image_server_port 5555
```

(You can also run normal eval_g1)

Find robot interface. Run this line where you run the eval code:
```bash
ip -br addr | grep 192.168.123.164
```
(Normally is `eth0`)

Find computer IP for `--policy_host` on robot side. Run this line where you run the policy server:
```bash
ip -br addr
```

Pick the Ethernet IP on the same subnet (typically `192.168.123.222` or `192.168.123.x`). If you are using wifi, choose the wifi IP (typically `10.10.0.139`)

## Notes

- `run.sh` uses `--network host` so DDS and policy/image sockets can talk to robot/network services.
- If policy server is remote (your computer), `--policy_host` must be your computer IP, not `127.0.0.1`.
- The image installs only runtime dependencies required by the two eval scripts and the local G1 robot SDK path.