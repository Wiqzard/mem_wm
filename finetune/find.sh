#!/bin/bash

video_output="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/videos_gen_new.txt"
image_output="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/images_gen_new.txt"

# Clear existing output files
> "$video_output"
> "$image_output"

find /capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/cache/video_latent/cogvideox1.5-i2v-wm/ -type f -name "*.safetensors" | while read safetensors_file; do
    base_name="$(basename "$safetensors_file" .safetensors)"

    # Video path
    mp4_file="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/videos/${base_name}.mp4"
    if [[ -f "$mp4_file" ]]; then
        echo "$mp4_file" >> "$video_output"
    fi

    # Image path
    png_file="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/first_frames/${base_name}.png"
    if [[ -f "$png_file" ]]; then
        echo "$png_file" >> "$image_output"
    fi
done

echo "Generated $video_output and $image_output."

#find /capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/cache/video_latent/cogvideox1.5-i2v-wm/ -type f -name "*.safetensors" | while read safetensors_file; do
#    mp4_filename="$(basename "$safetensors_file" .safetensors).mp4"
#    mp4_file="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/videos/$mp4_filename"
#    
#    if [[ -f "$mp4_file" ]]; then
#        echo "$mp4_file"
#    fi
#done > /capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/videos_gen_2.txt

