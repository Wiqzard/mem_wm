import os
import json
import random
from tqdm import tqdm
import imageio.v3 as iio  # Use imageio's latest API

def main():
    # Parameters
    root_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set"
    videos_dir = os.path.join(root_dir, "videos")
    metadata_dir = os.path.join(root_dir, "metadata")
    first_frames_dir = os.path.join(root_dir, "first_frames")
    n_actions = 82  # We only proceed if actions < n_actions

    # Create the 'first_frames' directory if it doesn't exist
    os.makedirs(first_frames_dir, exist_ok=True)

    videos_list = []
    images_list = []

    # Iterate over all mp4 files in the videos directory
    for video_file in tqdm(os.listdir(videos_dir)):
        if video_file.endswith(".mp4"):
            base_name = os.path.splitext(video_file)[0]
            metadata_file = base_name + ".json"
            metadata_path = os.path.join(metadata_dir, metadata_file)

            # 1. Check for corresponding .json
            if not os.path.exists(metadata_path):
                continue

            # 2. Check if the JSON has "actions" and if it's less than n_actions
            with open(metadata_path, "r") as f:
                data = json.load(f)

            if "actions" not in data:
                continue

            actions_count = len(data["actions"])
            if actions_count <= n_actions:
                continue

            # 3. If all checks are good, extract the first frame using imageio
            video_path = os.path.join(videos_dir, video_file)
            try:
                first_frame = iio.imread(video_path, index=0)  # Read first frame
            except Exception as e:
                print(f"Error reading {video_file}: {e}")
                continue

            # Construct output path for the first frame
            image_name = base_name + ".png"
            image_path = os.path.join(first_frames_dir, image_name)
            
            try:
                iio.imwrite(image_path, first_frame)
                videos_list.append(f"videos/{video_file}")
                images_list.append(f"first_frames/{image_name}")
                print(f"Processed {video_file} and saved {image_name}.")
            except Exception as e:
                print(f"Error saving {image_name}: {e}")
                continue

    # Write all valid video paths to videos.txt and image paths to images.txt
    videos_txt_path = os.path.join(root_dir, "videos.txt")
    images_txt_path = os.path.join(root_dir, "images.txt")

    with open(videos_txt_path, "w") as f_v, open(images_txt_path, "w") as f_i:
        for v, i in zip(videos_list, images_list):
            f_v.write(v + "\n")
            f_i.write(i + "\n")

    # Randomly sample 100 (or fewer if there aren't enough) entries 
    combined = list(zip(videos_list, images_list))
    random.shuffle(combined)
    sample = combined[:100]

    videos_100_list = [v for v, _ in sample]
    images_100_list = [i for _, i in sample]

    videos_100_txt_path = os.path.join(root_dir, "videos_100.txt")
    images_100_txt_path = os.path.join(root_dir, "images_100.txt")

    with open(videos_100_txt_path, "w") as f_v, open(images_100_txt_path, "w") as f_i:
        for v, i in zip(videos_100_list, images_100_list):
            f_v.write(v + "\n")
            f_i.write(i + "\n")

    print("Done! Check videos.txt, images.txt, videos_100.txt, and images_100.txt.")

if __name__ == "__main__":
    main()