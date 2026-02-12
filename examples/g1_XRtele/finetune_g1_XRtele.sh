
set -x -e

export NUM_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
#     gr00t/experiment/launch_finetune.py \
#     --base_model_path  nvidia/GR00T-N1.6-3B \
#     --dataset_path /home/rss/.cache/huggingface/lerobot/rss-hiverobots/grt_pick_multiple_toys_21jan_1cam_prompt2_id \
#     --modality_config_path examples/g1_XRtele/modality_config.py \
#     --embodiment_tag NEW_EMBODIMENT \
#     --num_gpus $NUM_GPUS \
#     --output_dir /mnt/sata1/gr00t16/g1_finetune/pick_multiple_toys_1cam_prompt2__lr1e-5 \
#     --save_total_limit 5 \
#     --max_steps 20000 \
#     --save_steps 5000 \
#     --warmup_ratio 0.05 \
#     --weight_decay 1e-5 \
#     --learning_rate 1e-5 \
#     --global_batch_size 16 \
#     --gradient_accumulation_steps 32 \
#     --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
#     --num-shards-per-epoch 100 \
#     --dataloader-num-workers 4 \
#     --shard-size 75 \
#     --use_wandb \


torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-3B \
    --dataset_path /home/rss/.cache/huggingface/lerobot/rss-hiverobots/grt_pick_multiple_toys_21jan_1cam_prompt2_id \
    --modality_config_path examples/g1_XRtele/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /mnt/sata1/gr00t16/g1_finetune/pick_multiple_toys_1cam_prompt2__lr1e-4 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --save_steps 5000 \
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