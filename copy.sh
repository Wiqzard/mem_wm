#!/usr/bin/env bash

# Text file containing the list of .png paths
INPUT_FILE="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set/images_100a.txt"


# The directory where we want to copy the videos
DEST_DIR="./data/val_videos"

# Create the destination directory if it doesn't exist
mkdir -p "${DEST_DIR}"

# Read each line (image path) from the input file
while IFS= read -r png_path; do
    # Skip empty lines
    [[ -z "$png_path" ]] && continue
    
    # Replace 'first_frames' with 'videos' and change extension from .png to .mp4
    mp4_path="${png_path/first_frames/videos}"
    mp4_path="${mp4_path%.png}.mp4"
    
    # Copy the .mp4 file to the destination directory
    echo "Copying: ${mp4_path} -> ${DEST_DIR}"
    cp "${mp4_path}" "${DEST_DIR}" 2>/dev/null || {
      echo "Warning: Could not copy ${mp4_path}. File may not exist."
    }
done < "${INPUT_FILE}"

echo "Done copying videos to ${DEST_DIR}."
