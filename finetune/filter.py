import os
import json
import logging
from safetensors.torch import load_file
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(processName)s: %(message)s')

# Directory where the .safetensors files are located
SAFETENSORS_DIR = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set/cache/video_latent/cogvideox1.5-i2v-wm/81x368x640"

def process_video(video_info):
    """Process a single video, checking its action count from JSON metadata 
    and verifying safetensors shape."""
    video_path, image_path, n_actions = video_info

    # Extract filename without extension
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]

    # Construct the metadata JSON path
    json_path = video_path.replace("/videos/", "/metadata/").replace(".mp4", ".json")

    # If JSON file does not exist, skip
    if not os.path.isfile(json_path):
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

    # Now check the corresponding .safetensors file
    safetensors_path = os.path.join(SAFETENSORS_DIR, f"{name_no_ext}.safetensors")
    if not os.path.isfile(safetensors_path):
        logging.info(f"safetensors file not found for video: {video_path}")
        return None

    # Load the safetensors file
    st_data = load_file(safetensors_path)
    # In case there's more than one key, just pick the first
    first_key = list(st_data.keys())[0]
    tensor = st_data[first_key]

    # Check if the shape matches (16, 21, 46, 80)
    expected_shape = (16, 20, 46, 80)
    if tensor.shape != expected_shape:
        logging.info(
            f"Invalid safetensors shape for video '{video_path}': "
            f"{tensor.shape} (expected {expected_shape})"
        )
        return None

    # If we reach here, the video passes both checks
    return video_path, image_path

def main():
    base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
    videos_input_path = os.path.join(base_dir, "videos_gen_new.txt")
    images_input_path = os.path.join(base_dir, "images_gen_new.txt")
    videos_output_path = os.path.join(base_dir, "videos_filtered3.txt")
    images_output_path = os.path.join(base_dir, "images_filtered3.txt")

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
    video_info_list = [(video_lines[i], image_lines[i], n_actions) for i in range(len(video_lines))]

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
#import os
#import json
#from tqdm import tqdm
#from concurrent.futures import ProcessPoolExecutor
#import logging
#logging.basicConfig(level=logging.INFO, format='%(processName)s: %(message)s')


#def process_video(video_info):
    #"""Process a single video, checking its action count from JSON metadata."""
    #video_path, image_path, n_actions = video_info

    ## Extract filename without extension
    #filename = os.path.basename(video_path)
    #name_no_ext = os.path.splitext(filename)[0]

    ## Construct the metadata JSON path
    #json_path = video_path.replace("/videos/", "/metadata/").replace(".mp4", ".json")

    ## If JSON file does not exist, skip
    #if not os.path.isfile(json_path):
        #logging.info(f"JSON file not found for video: {video_path}")
        #return None

    ## Read and parse JSON file
    #try:
        #with open(json_path, "r") as jfile:
            #data = json.load(jfile)
    #except json.JSONDecodeError:
        #return None

    ## Count actions
    #if "actions" not in data:
        #logging.info(f"Invalid actions data for video: {video_path}")
        #return None
    
    #actions_count = len(data["actions"])

    ## Filter based on threshold
    #if actions_count >= n_actions:
        #return video_path, image_path
    #logging.info(f"Skipping video: {video_path} with {actions_count} actions.")
    #return None

#def main():
    #base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set"
    #videos_input_path = os.path.join(base_dir, "videos_gen_new.txt")
    #images_input_path = os.path.join(base_dir, "images_gen_new.txt")
    #videos_output_path = os.path.join(base_dir, "videos_filtered2.txt")
    #images_output_path = os.path.join(base_dir, "images_filtered2.txt")

    ## Number of actions threshold
    #n_actions = 90  

    ## Read input files
    #with open(videos_input_path, "r") as vf:
        #video_lines = [line.strip() for line in vf if line.strip()]
    
    #with open(images_input_path, "r") as inf:
        #image_lines = [line.strip() for line in inf if line.strip()]
    
    #if len(video_lines) != len(image_lines):
        #raise ValueError("Mismatch in number of lines between videos and images input files.")

    ## Prepare input for parallel processing
    #video_info_list = [(video_lines[i], image_lines[i], n_actions) for i in range(len(video_lines))]

    ## Use multiprocessing to parallelize processing
    #filtered_videos = []
    #filtered_images = []
    #with ProcessPoolExecutor() as executor:
        #results = list(tqdm(executor.map(process_video, video_info_list), total=len(video_info_list)))

    ## Collect results
    #for result in results:
        #if result:
            #filtered_videos.append(result[0])
            #filtered_images.append(result[1])
        #else:
            #print("Skipping invalid video.")

    ## Write results to files
    #with open(videos_output_path, "w") as vf:
        #vf.write("\n".join(filtered_videos) + "\n")

    #with open(images_output_path, "w") as inf:
        #inf.write("\n".join(filtered_images) + "\n")

    #print(f"Done. Kept {len(filtered_videos)} videos/images with >= {n_actions} actions.")

#if __name__ == "__main__":
    #main()