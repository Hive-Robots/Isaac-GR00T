
set -x -e

export NUM_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-3B \
    --dataset_path /home/rss/.cache/huggingface/lerobot/rss-hiverobots/gr00t16_test \
    --modality_config_path examples/g1_XRtele/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /tmp/g1_finetune \
    --save_total_limit 5 \
    --max_steps 10 \
    --save_steps 10 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --num-shards-per-epoch 100 \
    --dataloader-num-workers 4 \
    --shard-size 75 \
    --use_wandb \
