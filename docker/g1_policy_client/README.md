# G1 Policy Client Container

Minimal Docker image for running:
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

Or run eval directly:

```bash
bash run.sh python gr00t/eval/real_robot/g1/eval_g1.py \
  --modality_config_path examples/g1_XRtele/modality_config.py \
  --modality_config_name unitree_g1_xrtele \
  --policy_host 127.0.0.1 \
  --policy_port 5555 \
  --action_horizon 8 \
  --control_hz 25 \
  --network_interface enx9c69d31ecd9b \
  --image_server_address 192.168.123.164 \
  --image_server_port 5555
```

Loop variant:

```bash
bash run.sh python gr00t/eval/real_robot/g1/eval_g1_loop.py \
  --modality_config_path examples/g1_XRtele/modality_config.py \
  --modality_config_name unitree_g1_xrtele \
  --policy_host 127.0.0.1 \
  --policy_port 5555 \
  --action_horizon 8 \
  --control_hz 25 \
  --network_interface enx9c69d31ecd9b \
  --image_server_address 192.168.123.164 \
  --image_server_port 5555
```

## Notes

- `run.sh` uses `--network host` so DDS and policy/image sockets can talk to robot/network services.
- The image installs only runtime dependencies required by the two eval scripts and the local G1 robot SDK path.
