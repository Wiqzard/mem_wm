#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/capstor/store/cscs/swissai/a03/hf
export  WANDB_API_KEY=11b574cdfa34332326c4a0a1ac8f7b06fb123637 #11b574cdfa34332326c4a0a1ac8f7b06fb123637

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B-I2V"
    #--model_name  "cogvideox-i2v-wm"
    --model_name "cogvideox1.5-i2v-wm"  # ["cogvideox-i2v"]
    #--model_type "i2v"
    --model_type "wm"
    --training_type "sft"
    #--training_type "lora"
    #--encoder_path 
    --local_path /capstor/scratch/cscs/sstapf/mem_wm/outputs/transformer
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "outputs"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
    --caption_column "prompts.txt"
    #--image_column "image.txt"
    #--video_column "videos_gen_2.txt" #"videos_matching.txt"
    --image_column "images_filtered3.txt"
    --video_column "videos_filtered3.txt" #"videos_matching.txt"
    #--image_column "images_gen_new_debug.txt"
    #--video_column "videos_gen_new_debug.txt" #"videos_matching.txt"
    --train_resolution "81x368x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x352x608"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x352x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x36x640"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "81x360x640"  # (frames x height x width), frames should be 8N+1
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    #--train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
    #--train_resolution "49x240x360"  # (frames x height x width), frames should be 8N+1

)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 100 # number of training epochs
    --seed 42 # random seed
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    #--learning_rate 2e-5
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory False #True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 50 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
  #  --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true #true #false  # ["true", "false"]
    --validation_dir "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set" #"/home/ss24m050/Documents/CogVideo/data/data_269"
    --validation_steps 50  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images_100.txt"
    --validation_videos "videos_100.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
#accelerate launch train.py \
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
