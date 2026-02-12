# Main Scripts and CLI Arguments

This page lists the primary scripts you run in this repo for finetuning and inference, along with all available CLI arguments and defaults.

## Finetuning (single-node)
Script: `gr00t/experiment/launch_finetune.py`
Run (example):
```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path demo_data/cube_to_bowl_5 \
  --embodiment-tag SO100 \
  --output-dir /tmp/gr00t_finetune
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


## Open-loop eval (offline, dataset-driven)
Script: `gr00t/eval/open_loop_eval.py`
Run (example):
```bash
uv run python gr00t/eval/open_loop_eval.py \
  --model-path /tmp/gr00t_finetune/checkpoint-10000 \
  --dataset-path demo_data/cube_to_bowl_5/ \
  --embodiment-tag NEW_EMBODIMENT
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


## Inference (standalone / local)
Script: `scripts/deployment/standalone_inference_script.py`
Run (example):

```bash
uv run python scripts/deployment/standalone_inference_script.py \
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

## Inference client-server
### Server
Script: `gr00t/eval/run_gr00t_server.py`
Run (example):
```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag GR1 \
  --model-path nvidia/GR00T-N1.6-3B \
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

### Client-side rollout against server or local model
Script: `gr00t/eval/rollout_policy.py`
Run (example, client against server):
```bash
uv run python gr00t/eval/rollout_policy.py \
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

Notes:
- These CLIs are defined via `tyro` dataclass parsing or `argparse`. Run `--help` for tyro/argparse output.
- For server-client inference in your own code, use `PolicyClient` from `gr00t/policy/server_client.py`.

## Running In Sim vs Real
**Simulation (server + client rollout)**
1. Start the policy server with your checkpoint:
```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag GR1 \
  --model-path /tmp/gr00t_finetune/checkpoint-10000 \
  --host 0.0.0.0 \
  --port 5555
```
2. Run a simulation client that connects to the server (example rollout entry point):
```bash
uv run python gr00t/eval/rollout_policy.py \
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
