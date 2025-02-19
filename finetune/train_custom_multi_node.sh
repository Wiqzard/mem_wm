#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/capstor/store/cscs/swissai/a03/hf
export  WANDB_API_KEY=11b574cdfa34332326c4a0a1ac8f7b06fb123637 #11b574cdfa34332326c4a0a1ac8f7b06fb123637

# Model Configuration
MODEL_ARGS=(
    --model_path  "THUDM/CogVideoX-2b"
    --model_name  "cogvideox-i2v-wm"
    #--model_path "THUDM/CogVideoX1.5-5B-I2V"
    #--model_name "cogvideox1.5-i2v-wm"  # ["cogvideox-i2v"]
    #--model_type "i2v"
    --model_type "wm"
    --training_type "sft"
    #--training_type "lora"
    #--encoder_path 
    #--local_path /capstor/scratch/cscs/sstapf/mem_wm/outputs/transformer_5b
    #--local_path /capstor/scratch/cscs/sstapf/mem_wm/outputs/transformer_2b_iv_grp_2
    #--local_path /capstor/scratch/cscs/sstapf/mem_wm/outputs/transformer_actions
    --local_path /capstor/scratch/cscs/sstapf/mem_wm/finetune/outputs/basalt_10_fps_actions_full_res/checkpoint-6000

    #--local_path /capstor/scratch/cscs/sstapf/mem_wm/finetune/outputs/training_hlr_49_grp_cont/checkpoint-2400 
)

# Output Configuration
OUTPUT_ARGS=(
   #--output_dir "outputs/outputs_2_hlr_81_cached_fp16"
    --output_dir "outputs/basalt_10_fps_actions_full_res_gf_cont" #outputs_2_hlr_49_cont"
    #--output_dir "outputs/outputs_1.5_hlr_cont"
    #--output_dir "outputs/outputs_2_hlr_49_fp16"
    #--output_dir "outputs/outputs_1.5_hlr"
    --report_to "wandb"
    #--tracker_name "gem_test_2"
)

# Data Configuration
DATA_ARGS=(
    #--data_root "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed2/train_set"
    #--caption_column "prompts.txt"
    #--image_column "images_train_combined.txt"
    #--video_column "videos_train_combined.txt"

    --data_root "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed/train_set"
    --caption_column "prompts.txt"
    --image_column "images_train_gf_processed.txt"
    --video_column "videos_train_gf_processed.txt"


    #--image_column "images_train_14.txt"
    #--video_column "videos_train_14.txt"

    #--image_column "images.txt"
    #--video_column "videos.txt" #"videos_matching.txt"
    #--image_column "images_filtered4.txt"
    #--video_column "videos_filtered4.txt" #"videos_matching.txt"
    #--image_column "images_gen_new_debug.txt"
    #--video_column "videos_gen_new_debug.txt" #"videos_matching.txt"
    #--train_resolution "81x368x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "81x352x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "25x352x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x352x640"  # (frames x height x width), frames should be 8N+1
    --train_resolution "25x128x232"  
    --encode_online 1

)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 1000 # number of training epochs
    --seed 42 # random seed
    --batch_size  32 #4 #16 #4 #16
    --gradient_accumulation_steps 1
    --mixed_precision "bf16" #"fp16" #"bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    #--learning_rate 0.00006 #2e-5
    --learning_rate 0.0001 #5 #1 #1 #8 #2e-5
    --lr_scheduler cosine
    --lr_warmup_steps 500
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory False #True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 200 #100 #2 #200 #100 # save checkpoint every x steps
    --checkpointing_limit 8 # maximum number of checkpoints to keep, after which the oldest one is deleted
  #  --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true #true #false  # ["true", "false"]
    --validation_dir "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed2/test_set" #"/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set" #"/home/ss24m050/Documents/CogVideo/data/data_269"
    --validation_steps 200 #5#2 #200 #100  # should be multiple of checkpointing_steps
    --validation_prompts "prompts_test_combined.txt"
    --validation_images "images_test_combined.txt"
    --validation_videos "videos_test_combined.txt"
    #--validation_prompts "prompts_test_14.txt"
    #--validation_images "images_test_14.txt"
    #--validation_videos "videos_test_14.txt"
    #--validation_prompts "prompts_test_gf_processed.txt"
    #--validation_images "images_test_gf_processed.txt"
    #--validation_videos "videos_test_gf_processed.txt"
    --gen_fps 16 #0
)

ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MAIN_ADDR=$master_addr
#MAIN_ADDR=$(echo "${SLURM_NODELIST}" | sed 's/[],].*//g; s/\[//g')
MAIN_PORT=12852

echo "Number of nodes: $SLURM_NNODES"
echo "Number of processes: $ACCEL_PROCS"
echo "Main process address: $MAIN_ADDR"
echo "Main process port: $MAIN_PORT"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_NODEID=$SLURM_NODEID"



# Combine all arguments and launch training
#accelerate launch train.py \
accelerate launch --config_file accelerate_config.yaml \
    --num_machines=$SLURM_NNODES \
    --num_processes=$ACCEL_PROCS \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MAIN_ADDR \
    --main_process_port $MAIN_PORT \
    train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
