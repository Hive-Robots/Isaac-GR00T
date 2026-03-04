# Main Scripts and CLI Arguments

This page lists the primary scripts you run in this repo for finetuning and inference, along with all available CLI arguments and defaults.

All scripts are meant to be run from:

```bash
cd ~/Isaac-GR00T
conda activate gr00t
```

## Finetuning (single-node)
Script: `gr00t/experiment/launch_finetune.py`
Run (example):
```bash
python gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --dataset_path /home/rss/.cache/huggingface/lerobot/rss-hiverobots/grt_pick_multiple_toys_21jan_1cam_prompt2_id \
    --modality_config_path examples/g1_XRtele/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus 1 \
    --output_dir /mnt/sata1/gr00t16/g1_finetune/pick_toys_1cam_prompt2__bs32_lr1e4_shxep10000_g1 \
    --max_steps 10000 \
    --save_steps 1000 \
    --save-total-limit 3 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --num-shards-per-epoch 10000 \
    --dataloader-num-workers 4 \
    --shard-size 75 \
    --use_wandb \
```
Arguments:
- `--base-model-path` (str, required). Path to pretrained base model checkpoint (Hugging Face ID or local dir).
- `--dataset-path` (str, required). Dataset root directory containing trajectory data.
- `--embodiment-tag` (EmbodimentTag, required). Embodiment identifier (e.g., `GR1`, `SO100`, `NEW_EMBODIMENT`).
- `--modality-config-path` (str | None, default: None). Path to a Python file defining the modality configuration.
- `--tune-llm` (bool, default: False). Fine-tune the language model backbone.
- `--tune-visual` (bool, default: False). Fine-tune the visual encoder.
- `--tune-projector` (bool, default: True). Fine-tune multimodal projector layers.
- `--tune-diffusion-model` (bool, default: True). Fine-tune diffusion-based action decoder.
- `--state-dropout-prob` (float, default: 0.0). Dropout probability for state inputs.
- `--random-rotation-angle` (int | None, default: None). Max rotation angle in degrees for augmentation.
- `--color-jitter-params` (dict[str, float] | None, default: None). Color jitter params; expected keys include `brightness`, `contrast`, `saturation`, `hue`.
- `--global-batch-size` (int, default: 64). Total effective batch size.
- `--dataloader-num-workers` (int, default: 2). Data loader workers.
- `--learning-rate` (float, default: 1e-4). Initial learning rate.
- `--gradient-accumulation-steps` (int, default: 1). Accumulation steps.
- `--output-dir` (str, default: `./outputs`). Output directory for checkpoints/logs.
- `--save-steps` (int, default: 1000). Checkpoint save frequency.
- `--save-total-limit` (int, default: 5). Max checkpoints kept.
- `--num-gpus` (int, default: 1). Number of GPUs.
- `--use-wandb` (bool, default: False). Enable Weights & Biases logging.
- `--max-steps` (int, default: 10000). Total training steps.
- `--weight-decay` (float, default: 1e-5). Weight decay.
- `--warmup-ratio` (float, default: 0.05). LR warmup ratio.
- `--shard-size` (int, default: 1024). Shard size for dataset preloading.
- `--episode-sampling-rate` (float, default: 0.1). Episode sampling rate.
- `--num-shards-per-epoch` (int, default: 100000). Number of shards per epoch.

Or train using a bash file:

```bash
bash examples/g1/finetune_g1_XRtele.sh
```

## Open-loop eval (offline, dataset-driven)
Script: `gr00t/eval/open_loop_eval.py`
Run (example):
```bash
python gr00t/eval/open_loop_eval.py \
  --model-path /tmp/gr00t_finetune/checkpoint-10000 \
  --dataset-path demo_data/cube_to_bowl_5/ \
  --embodiment-tag NEW_EMBODIMENT \
  --save-plot-path /mnt/sata1/gr00t16/open-loop-eval/
```
Arguments:
- `--host` (str, default: `127.0.0.1`). Host to connect to (used only if `--model-path` is not provided).
- `--port` (int, default: 5555). Port to connect to (used only if `--model-path` is not provided).
- `--steps` (int, default: 200). Max steps per trajectory.
- `--traj-ids` (list[int], default: `[0]`). Trajectory IDs to evaluate.
- `--action-horizon` (int, default: 16). Action horizon.
- `--dataset-path` (str, default: `demo_data/cube_to_bowl_5/`). Dataset path.
- `--embodiment-tag` (EmbodimentTag, default: `NEW_EMBODIMENT`). Embodiment tag.
- `--model-path` (str | None, default: None). Model checkpoint path. If omitted, uses PolicyClient with `--host/--port`.
- `--denoising-steps` (int, default: 4). Number of denoising steps. (Not used in this script!)
- `--save-plot-path` (str | None, default: None). Path to save plot. By default saves at /mnt/open_loop_eval
- `--modality-keys` (list[str] | None, default: None). Modality keys to plot (if None, plots all).

## Inference client-server
### Server
Script: `gr00t/eval/run_gr00t_server.py`
Run (example):
```bash
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/g1_XRtele/modality_config.py\
  --model-path /mnt/sata1/gr00t16/... \
  --host 0.0.0.0 \
  --port 5555
```
Arguments:
- `--model-path` (str | None, default: None). Model checkpoint path. Required unless using `--dataset-path`.
- `--embodiment-tag` (EmbodimentTag, default: `NEW_EMBODIMENT`). Embodiment tag.
- `--device` (str, default: `cuda`). Device string (e.g., `cuda`, `cuda:0`, `cpu`).
- `--dataset-path` (str | None, default: None). Dataset path for replay mode.
- `--modality-config-path` (str | None, default: None). Modality config file (JSON) for replay mode.
- `--execution-horizon` (int | None, default: None). Policy execution horizon for replay mode.
- `--host` (str, default: `0.0.0.0`). Host address.
- `--port` (int, default: 5555). Port.
- `--strict` (bool, default: True). Enforce strict input/output validation.
- `--use-sim-policy-wrapper` (bool, default: False). Wrap policy with sim wrapper.

If you are debugging, it is recommended to use the `dataset-path` argument instead of the `model-path`. That way it will replay a recorded episode rather than calculating actions. If running replay episode, it is important to use the `execution-horizon` arg.

### Client-side rollout against server or local model
Use a robot-specific inference script:

```bash
python gr00t/eval/real_robot/g1/eval_g1_loop.py \
  --modality_config_path examples/g1_XRtele/modality_config.py \
  --modality_config_name unitree_g1_xrtele \
  --policy_host 127.0.0.1 \
  --policy_port 5555 \
  --action_horizon 16 \
  --control_hz 25 \
  --network_interface enx9c69d31ecd9b \
  --record true \
  --record_save_dir ./g1_eval_records
```
Arguments:
- `--modality_config_path` (str, default: `examples/g1_XRtele/modality_config.py`). Path to modality config Python file.
- `--modality_config_name` (str, default: `unitree_g1_xrtele`). Modality config object name inside the config module.
- `--policy_host` (str, default: `localhost`). Policy server host.
- `--policy_port` (int, default: 5555). Policy server port.
- `--action_horizon` (int, default: 8). Number of predicted actions to request per policy inference.
- `--lang_instruction` (str, default: `Perform the task.`). Language instruction passed to the policy.
- `--control_hz` (float, default: 25.0). Main control loop frequency.
- `--state_init_timeout_s` (float, default: 5.0). Timeout waiting for initial robot state.
- `--camera_init_timeout_s` (float, default: 5.0). Timeout waiting for camera frames.
- `--require_start_keypress` (bool, default: True). Require typing `s` before starting robot motion/eval loop.
- `--image_server_address` (str, default: `192.168.123.164`). Image server address.
- `--image_server_port` (int, default: 5555). Image server port.
- `--head_image_height` (int, default: 480). Head camera image height.
- `--head_image_width` (int, default: 640). Head camera image width.
- `--wrist_image_height` (int, default: 480). Wrist camera image height.
- `--wrist_image_width` (int, default: 640). Wrist camera image width.
- `--arm` (str, default: `G1_29`). Robot arm configuration key used by `robot_sdk`.
- `--ee` (str, default: `dex3`). End-effector configuration key used by `robot_sdk`.
- `--ee_side` (str, default: `both`). End-effector side selection.
- `--sim` (bool, default: False). Enable simulation mode in robot setup path.
- `--motion` (bool, default: False). Motion flag forwarded to robot setup path.
- `--network_interface` (str | None, default: None). Optional DDS network interface name (autodetect when omitted).
- `--enable_diagnostics` (bool, default: False). Write loop diagnostics CSV.
- `--dataset_path` (str | None, default: None). Optional dataset path for loading initial robot pose.
- `--record` (bool, default: False). Enable episode recording during loop eval.
- `--record_dir` (str, default: `g1_eval_records`). Default recording output directory.
- `--record_save_dir` (str | None, default: None). Explicit recording output directory; when set, takes precedence over `--record_dir`.
- `--record_rerun_log` (bool, default: False). Enable rerun logging in `EpisodeWriter` while recording.

You can check [`docker/g1_policy_client`](../docker/g1_policy_client/README.md) for more information on running the `eval_g1` code directly on the robot.

## Other Inference (Local &/or simulation)

To run a simulated environment (Only works with baseline examples):
```bash
python gr00t/eval/rollout_policy.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
```
Arguments:
- `--max-episode-steps` (int, default: 504). Max steps per episode.
- `--n-episodes` (int, default: 50). Number of episodes to run.
- `--model-path` (str, default: ""). Local model path for direct inference (mutually exclusive with client args).
- `--policy-client-host` (str, default: ""). Policy server host (use with `--policy-client-port`).
- `--policy-client-port` (int | None, default: None). Policy server port.
- `--env-name` (str, default: `gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env`). Env name.
- `--n-envs` (int, default: 8). Number of parallel environments.
- `--n-action-steps` (int, default: 8). Action steps per policy call.

### Inference (standalone / local)
! Not recommended! It is better to use client-server even if they are within the same pc.
Script: `scripts/deployment/standalone_inference_script.py`
Run (example):

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path /tmp/gr00t_finetune/checkpoint-10000 \
  --dataset-path demo_data/robot_sim.PickNPlace/ \
  --embodiment-tag GR1
```

Arguments:
- `--host` (str, default: `127.0.0.1`). Host to connect to. (Not used in this script!)
- `--port` (int, default: 5555). Port to connect to. (Not used in this script!)
- `--steps` (int, default: 200). Max steps per trajectory.
- `--traj-ids` (list[int], default: `[0]`). Trajectory IDs to evaluate.
- `--action-horizon` (int, default: 16). Action horizon.
- `--video-backend` (str, default: `torchcodec`). One of `decord`, `torchvision_av`, `torchcodec`.
- `--dataset-path` (str, default: `demo_data/robot_sim.PickNPlace/`). Dataset path.
- `--embodiment-tag` (EmbodimentTag, default: `GR1`). Embodiment tag.
- `--model-path` (str | None, default: None). Model checkpoint path.
- `--inference-mode` (str, default: `pytorch`). One of `pytorch`, `tensorrt`.
- `--trt-engine-path` (str, default: `./groot_n1d6_onnx/dit_model_bf16.trt`). TensorRT engine path, used when `--inference-mode tensorrt`.
- `--denoising-steps` (int, default: 4). Number of denoising steps. (Not used in this script!)
- `--save-plot-path` (str | None, default: None). Path to save plot. (Not used in this script!)
- `--skip-timing-steps` (int, default: 1). Skip initial steps for timing stats.
- `--get-performance-stats` (bool, default: True). Aggregate and summarize timing/accuracy stats.
- `--seed` (int, default: 42). Random seed.


Notes:
- These CLIs are defined via `tyro` dataclass parsing or `argparse`. Run `--help` for tyro/argparse output.
- For server-client inference in your own code, use `PolicyClient` from `gr00t/policy/server_client.py`.

## Dataset conversion (`convert_unitree_to_v2`)

It is necessary to convert the scripts from xr_teleoperate to Lerobot V2.1 in order to be trained in gr00t1.6. You can do so with the following script:

Run (example):
```bash
python scripts/lerobot_conversion/convert_unitree_to_v2.py \
  --raw-dir /path/to/xr_teleoperate_dataset \
  --repo-id your_name/your_dataset \
  --robot-type Unitree_G1_Dex3 \
  --mode video \
  --fps 25 \
  --push-to-hub
```
Arguments:
- `--raw-dir` (Path, required). Root directory of the Unitree XR-teleoperate dataset (`data.json` episodes + image files).
- `--repo-id` (str, required). Output LeRobot dataset id under `HF_LEROBOT_HOME` (format: `user_or_org/dataset_name`).
- `--robot-type` (str, required). Robot config key from `scripts/lerobot_conversion/constants.py` (for example `Unitree_G1_Dex3`: Head+ 2 wrist cams, or `Unitree_G1_Dex3_real`:Only head).
- `--modality-config-path` (Path | None, default: None). Optional JSON modality config to override the hardcoded robot config.
- `--push-to-hub` (bool, default: False). Upload converted dataset to Hugging Face Hub after conversion.
- `--mode` (`video` | `image`, default: `video`). For GR00T 1.6, only `video` is supported.
- `--fps` (float, default: 25.0). Output video frame rate; must be greater than 0.
- `--dataset-config.use-videos` (bool, default: True). Internal writer setting for video-backed datasets.
- `--dataset-config.tolerance-s` (float, default: `0.0001`). Timestamp tolerance used by the dataset writer config.
- `--dataset-config.image-writer-processes` (int, default: 10). Worker process count in dataset writer config.
- `--dataset-config.image-writer-threads` (int, default: 5). Worker thread count in dataset writer config.
- `--dataset-config.video-backend` (str | None, default: None). Optional backend override passed into dataset config.
- `--force-conversion` (bool, default: False). Reserved option (currently parsed but not used by conversion logic).

## Running In Sim vs Real
**Simulation (server + client rollout)**
1. Start the policy server with your checkpoint:
```bash
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag GR1 \
  --model-path /tmp/gr00t_finetune/checkpoint-10000 \
  --host 0.0.0.0 \
  --port 5555
```
2. Run a simulation client that connects to the server (example rollout entry point):
```bash
python gr00t/eval/rollout_policy.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
```
For benchmark-specific clients, follow the example READMEs under `examples/` (e.g., `examples/GR00T-WholeBodyControl/README.md`, `examples/BEHAVIOR/README.md`, `examples/robocasa-gr1-tabletop-tasks/README.md`).

**Real robot (server + PolicyClient integration)**
1. Start the policy server on a GPU machine as above.
2. On the robot control machine, use `PolicyClient` to send observations and receive actions. See the usage in `getting_started/policy.md` and the real-robot example `gr00t/eval/real_robot/SO100/eval_so100.py`.

**Why not `rollout_policy.py` for real robots?**
`gr00t/eval/rollout_policy.py` is designed for Gymnasium simulation environments in this repo. It expects a sim env name and uses sim wrappers, so it does not interface with real robot sensors/actuators. For real hardware, run the policy server and integrate `PolicyClient` in your robot control loop.
