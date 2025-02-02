import os
from tqdm import tqdm

# Define paths
base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
video_list_path = os.path.join(base_dir, "videos.txt")
metadata_folder = os.path.join(base_dir, "metadata")
output_file = os.path.join(base_dir, "videos_matching.txt")

# Read video file paths
with open(video_list_path, "r") as f:
    video_paths = [line.strip() for line in f]

# Check for matching JSON files with tqdm for progress tracking
matching_videos = []
for video_path in tqdm(video_paths, desc="Checking metadata", unit="file"):
    json_path = os.path.join(metadata_folder, os.path.basename(video_path).replace(".mp4", ".json"))
    if os.path.exists(json_path):
        matching_videos.append(video_path)

# Write matches to file
with open(output_file, "w") as f:
    for match in matching_videos:
        f.write(match + "\n")

print(f"Matching videos saved to {output_file}")
