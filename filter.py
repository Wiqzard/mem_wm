import os
import json
import logging
from safetensors.torch import load_file
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(processName)s: %(message)s')

# Directory where the .safetensors files are located
#SAFETENSORS_DIR = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/cache/video_latent/cogvideox1.5-i2v-wm/81x368x640"

def process_video(video_info):
    """Process a single video, checking its action count from JSON metadata 
    and verifying safetensors shape."""
    base_dir, video_path, image_path, n_actions = video_info

    # Extract filename without extension
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]

    # Construct the metadata JSON path
    json_path = video_path.replace("videos/", "metadata/").replace(".mp4", ".json")
    json_path = base_dir + "/" + json_path

    # If JSON file does not exist, skip
    if not os.path.isfile(json_path):
        #logging.info(f"Processing json: {json_path}")
        logging.info(f"JSON file not found for video: {video_path}")
        return None

    # Read and parse JSON file
    try:
        with open(json_path, "r") as jfile:
            data = json.load(jfile)
    except json.JSONDecodeError:
        logging.info(f"JSON decode error for: {json_path}")
        return None

    # Count actions
    if "actions" not in data:
        logging.info(f"Invalid actions data for video: {video_path}")
        return None
    
    actions_count = len(data["actions"])

    # Filter based on threshold
    if actions_count < n_actions:
        logging.info(f"Skipping video: {video_path} with {actions_count} actions.")
        return None

    ## Now check the corresponding .safetensors file
    #safetensors_path = os.path.join(SAFETENSORS_DIR, f"{name_no_ext}.safetensors")
    #if not os.path.isfile(safetensors_path):
    #    logging.info(f"safetensors file not found for video: {video_path}")
    #    return None

    ## Load the safetensors file
    #st_data = load_file(safetensors_path)
    ## In case there's more than one key, just pick the first
    #first_key = list(st_data.keys())[0]
    #tensor = st_data[first_key]

    ## Check if the shape matches (16, 21, 46, 80)
    #expected_shape = (16, 20, 46, 80)
    #if tensor.shape != expected_shape:
    #    logging.info(
    #        f"Invalid safetensors shape for video '{video_path}': "
    #        f"{tensor.shape} (expected {expected_shape})"
    #    )
    #    return None

    # If we reach here, the video passes both checks
    return video_path, image_path

def main():
    base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
    videos_input_path = os.path.join(base_dir, "videos.txt")
    images_input_path = os.path.join(base_dir, "images.txt")
    videos_output_path = os.path.join(base_dir, "videos_filtered4.txt")
    images_output_path = os.path.join(base_dir, "images_filtered4.txt")

    # Number of actions threshold
    n_actions = 90  

    # Read input files
    with open(videos_input_path, "r") as vf:
        video_lines = [line.strip() for line in vf if line.strip()]
    
    with open(images_input_path, "r") as inf:
        image_lines = [line.strip() for line in inf if line.strip()]
    
    if len(video_lines) != len(image_lines):
        raise ValueError("Mismatch in number of lines between videos and images input files.")

    # Prepare input for parallel processing
    video_info_list = [(base_dir, video_lines[i], image_lines[i], n_actions) for i in range(len(video_lines))]

    filtered_videos = []
    filtered_images = []

    # Use multiprocessing to parallelize processing
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_video, video_info_list), total=len(video_info_list)))

    # Collect results
    for result in results:
        if result:
            filtered_videos.append(result[0])
            filtered_images.append(result[1])
        else:
            # You can keep this print or convert it to a logging statement
            print("Skipping invalid video.")

    # Write results to files
    with open(videos_output_path, "w") as vf:
        vf.write("\n".join(filtered_videos) + "\n")

    with open(images_output_path, "w") as inf:
        inf.write("\n".join(filtered_images) + "\n")

    print(f"Done. Kept {len(filtered_videos)} videos/images with >= {n_actions} actions and valid safetensors shape.")

if __name__ == "__main__":
    main()


# Define paths
    base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"

#video_list_path = os.path.join(base_dir, "videos.txt")
#metadata_folder = os.path.join(base_dir, "metadata")
#output_file = os.path.join(base_dir, "videos_matching.txt")
#output_file_images = os.path.join(base_dir, "images_matching.txt")
#
## Read video file paths
#with open(video_list_path, "r") as f:
#    video_paths = [line.strip() for line in f]
#
## Check for matching JSON files with tqdm for progress tracking
#matching_videos = []
#for video_path in tqdm(video_paths, desc="Checking metadata", unit="file"):
#    json_path = os.path.join(metadata_folder, os.path.basename(video_path).replace(".mp4", ".json"))
#    if os.path.exists(json_path):
#        matching_videos.append(video_path)
#
## Write matches to file
#with open(output_file, "w") as f:
#    for match in matching_videos:
#        f.write(match + "\n")
#
#with open(output_file_images, "w") as f:
#    for match in matching_videos:
#        # replace videos/ with first_frames/ and .mp4 with .png
#        f.write(match.replace("videos/", "first_frames/").replace(".mp4", ".png") + "\n")
#
#print(f"Matching videos saved to {output_file}")
#
#
#import os
#from tqdm import tqdm
#
## Define paths
#base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
#video_list_path = os.path.join(base_dir, "videos.txt")
#metadata_folder = os.path.join(base_dir, "metadata")
#output_file = os.path.join(base_dir, "videos_matching.txt")
#output_file_images = os.path.join(base_dir, "images_matching.txt")
#
## Read video file paths
#with open(video_list_path, "r") as f:
#    video_paths = [line.strip() for line in f]
#
## Check for matching JSON files with tqdm for progress tracking
#matching_videos = []
#for video_path in tqdm(video_paths, desc="Checking metadata", unit="file"):
#    json_path = os.path.join(metadata_folder, os.path.basename(video_path).replace(".mp4", ".json"))
#    if os.path.exists(json_path):
#        matching_videos.append(video_path)
#
## Write matches to file
#with open(output_file, "w") as f:
#    for match in matching_videos:
#        f.write(match + "\n")
#
#with open(output_file_images, "w") as f:
#    for match in matching_videos:
#        # replace videos/ with first_frames/ and .mp4 with .png
#        f.write(match.replace("videos/", "first_frames/").replace(".mp4", ".png") + "\n")
#
#print(f"Matching videos saved to {output_file}")
#