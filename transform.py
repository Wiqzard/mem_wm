import os
import json
import math
import glob
import random
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

from moviepy.video.io.VideoFileClip import VideoFileClip


# append the base directory to videos/ metadata/ first_frames/ videos.txt prompts.txt images.txt
# in the end merge the txt files

################################################################################
# ADAPT THESE MAPPINGS FOR YOUR SUBFOLDERS AND THEIR PROMPTS
################################################################################
SUBFOLDER_PROMPTS = {
    "8": "The video depicts a scene within the Minecraft game environment where a player starts in a new world and builds a simple house...",
    "9": "In this video, the player uses the provided resources to build a house...",
    "10": "This video follows a player in a new Minecraft world as they attempt to craft a diamond pickaxe...",
    "11": "The video depicts a player searching for a cave in the Minecraft world...",
    "12": "Set in a mountainous Minecraft biome, the video shows a player with a water bucket...",
    "13": "The video takes place in a Minecraft village, where the player builds an animal pen...",
    "14": "In this video, the player spawns in a Minecraft village and uses the items in their inventory...",
}
################################################################################


def filter_action_fields(action_dict):
    """
    Filters out only the fields we need from each line of the .jsonl action.
    From 'keyboard', keep whether keys in [w,a,s,d,e,space,shift,ctrl,escape] are pressed.
    From 'mouse', keep dx, dy, and 'buttons'.
    """
    relevant_keys = {
        "key.keyboard.w": "w",
        "key.keyboard.a": "a",
        "key.keyboard.s": "s",
        "key.keyboard.d": "d",
        "key.keyboard.e": "e",
        "key.keyboard.space": "space",
        "key.keyboard.left.shift": "shift",
        "key.keyboard.right.shift": "shift",
        "key.keyboard.left.control": "ctrl",
        "key.keyboard.right.control": "ctrl",
        "key.keyboard.escape": "escape",
    }

    filtered = {}
    mouse_info = action_dict.get("mouse", {})
    filtered["dx"] = mouse_info.get("dx", 0.0)
    filtered["dy"] = mouse_info.get("dy", 0.0)
    filtered["buttons"] = mouse_info.get("buttons", [])
    filtered["dwheel"] = mouse_info.get("dwheel", 0.0)

    keyboard_info = action_dict.get("keyboard", {})
    pressed_keys = keyboard_info.get("keys", [])
    relevant_pressed = [relevant_keys[k] for k in pressed_keys if k in relevant_keys]

    filtered["keys"] = relevant_pressed
    return filtered


def read_and_filter_actions(jsonl_path):
    """
    Reads each line from the .jsonl file, parses it as JSON, and filters
    out only the relevant fields. Returns a list of filtered actions.
    """
    actions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            action_dict = json.loads(line)
            filtered = filter_action_fields(action_dict)
            actions.append(filtered)
    return actions

def split_video_and_actions(
    video_path,
    jsonl_path,
    output_dir,
    chunk_size,
    desired_fps=None,
    postfix = ""
):
    """
    Splits the video into chunks of size `chunk_size` frames (as .mp4).
    If `chunk_size` is -1, saves the full video without modification and extracts the first frame.
    Returns a list of (chunk_video_path, chunk_metadata_path, chunk_image_path).
    """
    
    # 1) Read & filter actions
    actions = read_and_filter_actions(jsonl_path)
    total_frames = len(actions) + 1

    # 2) Check video metadata with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames != total_video_frames:
        raise ValueError(
            f"Frame count mismatch for {os.path.basename(video_path)}:\n"
            f"  .jsonl lines: {total_frames} vs. Video frames: {total_video_frames}"
        )
    
    # 3) Handle chunking logic
    if chunk_size == -1:
        videos_out_dir = os.path.join(output_dir, "videos" + postfix)
        meta_out_dir = os.path.join(output_dir, "metadata" + postfix)
        frames_out_dir = os.path.join(output_dir, "first_frames" + postfix)

        os.makedirs(videos_out_dir, exist_ok=True)
        os.makedirs(meta_out_dir, exist_ok=True)
        os.makedirs(frames_out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        full_video_path = os.path.join(videos_out_dir, f"{base_name}_full.mp4")
        full_json_path = os.path.join(meta_out_dir, f"{base_name}_full.json")
        full_frame_path = os.path.join(frames_out_dir, f"{base_name}_full.png")
        
        shutil.copy(video_path, full_video_path)
        
        with open(full_json_path, 'w', encoding='utf-8') as jf:
            json.dump({"actions": actions}, jf, indent=2)
        
        # Extract first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(full_frame_path, frame)
        cap.release()
        
        return [(full_video_path, full_json_path, full_frame_path)]
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0 or -1")
    
    num_chunks = total_frames // chunk_size
    
    # 4) Prepare output dirs
    videos_out_dir = os.path.join(output_dir, "videos" + postfix)
    meta_out_dir = os.path.join(output_dir, "metadata" + postfix)
    frames_out_dir = os.path.join(output_dir, "first_frames" + postfix)

    os.makedirs(videos_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)
    os.makedirs(frames_out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_triples = []

    # 5) Use MoviePy to extract subclips
    video_clip = VideoFileClip(video_path)
    out_fps = desired_fps if desired_fps else original_fps

    for i in range(num_chunks):
        chunk_index = i + 1
        start_frame = i * chunk_size
        end_frame = min(start_frame + chunk_size, total_frames)

        chunk_actions = actions[start_frame:end_frame]

        # Convert frames -> seconds for subclip
        start_sec = start_frame / original_fps
        end_sec = end_frame / original_fps 

        # Extract subclip
        chunk_clip = video_clip.subclip(start_sec, end_sec)

        # Paths for output
        chunk_video_name = f"{base_name}_chunk_{chunk_index}.mp4"
        chunk_video_path = os.path.join(videos_out_dir, chunk_video_name)

        chunk_json_name = f"{base_name}_chunk_{chunk_index}.json"
        chunk_json_path = os.path.join(meta_out_dir, chunk_json_name)

        # Write subclip
        chunk_clip.write_videofile(
            chunk_video_path,
            fps=out_fps,
            audio=False,
            threads=1,
            logger=None
        )

        # Save metadata
        chunk_metadata = {"actions": chunk_actions}
        with open(chunk_json_path, 'w', encoding='utf-8') as jf:
            json.dump(chunk_metadata, jf, indent=2)

        # 6) Save the first frame of each chunk
        chunk_first_frame_name = chunk_video_name.replace(".mp4", ".png")
        chunk_first_frame_path = os.path.join(frames_out_dir, chunk_first_frame_name)
        chunk_clip.save_frame(chunk_first_frame_path, t=0)

        # Collect results
        result_triples.append((chunk_video_path, chunk_json_path, chunk_first_frame_path))

        # Clean up
        chunk_clip.close()
    
    video_clip.close()
    return result_triples


#def split_video_and_actions(
#    video_path,
#    jsonl_path,
#    output_dir,
#    chunk_size,
#    desired_fps=None,
#    postfix = ""
#):
#    """
#    Splits the video into chunks of size `chunk_size` frames (as .mp4).
#    Discards the last chunk if incomplete.
#
#    Returns a list of (chunk_video_path, chunk_metadata_path, chunk_image_path).
#    """
#    # 1) Read & filter actions
#    actions = read_and_filter_actions(jsonl_path)
#    total_frames = len(actions) + 1
#
#    # 2) Check video metadata with OpenCV
#    cap = cv2.VideoCapture(video_path)
#    if not cap.isOpened():
#        raise RuntimeError(f"Failed to open video: {video_path}")
#
#    original_fps = cap.get(cv2.CAP_PROP_FPS)
#    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#    cap.release()
#
#    if total_frames != total_video_frames:
#        raise ValueError(
#            f"Frame count mismatch for {os.path.basename(video_path)}:\n"
#            f"  .jsonl lines: {total_frames} vs. Video frames: {total_video_frames}"
#        )
#
#    # 3) Number of chunks (discard incomplete last chunk)
#
#    if chunk_size <= 0:
#        raise ValueError("chunk_size must be > 0")
#    num_chunks = total_frames // chunk_size
#
#    # 4) Prepare output dirs
#    videos_out_dir = os.path.join(output_dir, "videos" + postfix)
#    meta_out_dir = os.path.join(output_dir, "metadata" + postfix)
#    frames_out_dir = os.path.join(output_dir, "first_frames" + postfix)
#
#    os.makedirs(videos_out_dir, exist_ok=True)
#    os.makedirs(meta_out_dir, exist_ok=True)
#    os.makedirs(frames_out_dir, exist_ok=True)
#
#    base_name = os.path.splitext(os.path.basename(video_path))[0]
#
#    result_triples = []
#
#    # 5) Use MoviePy to extract subclips
#    video_clip = VideoFileClip(video_path)
#    out_fps = desired_fps if desired_fps else original_fps
#
#    for i in range(num_chunks):
#        chunk_index = i + 1
#        start_frame = i * chunk_size
#        end_frame = start_frame + chunk_size
#
#        chunk_actions = actions[start_frame:end_frame]
#
#        # Convert frames -> seconds for subclip
#        start_sec = start_frame / original_fps
#        end_sec = end_frame / original_fps 
#
#        # Extract subclip
#        chunk_clip = video_clip.subclip(start_sec, end_sec)
#
#        # Paths for output
#        chunk_video_name = f"{base_name}_chunk_{chunk_index}.mp4"
#        chunk_video_path = os.path.join(videos_out_dir, chunk_video_name)
#
#        chunk_json_name = f"{base_name}_chunk_{chunk_index}.json"
#        chunk_json_path = os.path.join(meta_out_dir, chunk_json_name)
#
#        # Write subclip
#        chunk_clip.write_videofile(
#            chunk_video_path,
#            fps=out_fps,
#            #codec="libx264",
#            #codec="mpeg4", #libx264",
#            audio=False,  # or True if needed
#            threads=1,
#            logger=None
#        )
#
#        # Save metadata
#        assert chunk_clip.duration == (end_sec - start_sec), f"Duration mismatch for chunk {chunk_index}"
#        assert len(chunk_actions) == chunk_size, f"Chunk {chunk_index} has {len(chunk_actions)} actions"
#
#        chunk_metadata = {"actions": chunk_actions}
#        with open(chunk_json_path, 'w', encoding='utf-8') as jf:
#            json.dump(chunk_metadata, jf, indent=2)
#
#        # 6) Save the first frame of each chunk
#        #    The first frame name must match the chunked video name but .png
#        chunk_first_frame_name = chunk_video_name.replace(".mp4", ".png")
#        chunk_first_frame_path = os.path.join(frames_out_dir, chunk_first_frame_name)
#
#        # t=0 grabs the first frame in that chunk
#        chunk_clip.save_frame(chunk_first_frame_path, t=0)
#
#        # Collect results
#        result_triples.append(
#            (chunk_video_path, chunk_json_path, chunk_first_frame_path)
#        )
#
#        # Clean up
#        chunk_clip.close()
#
#    video_clip.close()
#    return result_triples


def process_one_file(video_jsonl_prompt_tuple, output_dir, chunk_size, desired_fps, postfix):
    """
    Given a (video_path, jsonl_path, subfolder_prompt), splits into subclips.
    Returns a list of (chunk_video_path, prompt, chunk_image_path).
    """
    video_path, jsonl_path, subfolder_prompt = video_jsonl_prompt_tuple

    # split_video_and_actions returns:
    #   [ (chunk_video_path, chunk_metadata_path, chunk_image_path), ... ]
    triples = split_video_and_actions(
        video_path=video_path,
        jsonl_path=jsonl_path,
        output_dir=output_dir,
        chunk_size=chunk_size,
        desired_fps=desired_fps,
        postfix=postfix
    )

    # We only need to return the chunked video path, the prompt, and the image path
    results = []
    for (vid_path, _, img_path) in triples:
        results.append((vid_path, subfolder_prompt, img_path))

    return results


def find_subfolder_prompt(video_path, base_dir):
    """
    Finds which top-level folder (e.g. "8", "9") the video belongs to,
    returns the associated prompt.
    """
    rel_parts = os.path.relpath(video_path, base_dir).split(os.sep)
    subfolder = rel_parts[0]
    prompt = SUBFOLDER_PROMPTS.get(subfolder, "No prompt found")
    return prompt


def partition_dataset(video_jsonl_prompt_list, train_split, val_split, test_split, seed=42):
    """
    Shuffle and partition the dataset into train, val, (optionally test).
    Returns (train_list, val_list, test_list).
    """
    total = len(video_jsonl_prompt_list)
    if not (0 <= train_split <= 1 and 0 <= val_split <= 1 and 0 <= test_split <= 1):
        raise ValueError("Split percentages must be between 0 and 1.")
    if not math.isclose(train_split + val_split + test_split, 1.0, abs_tol=1e-7):
        raise ValueError("train_split + val_split + test_split must sum to 1.0")

    random.seed(seed)
    random.shuffle(video_jsonl_prompt_list)

    train_end = int(train_split * total)
    val_end = train_end + int(val_split * total)

    train_list = video_jsonl_prompt_list[:train_end]
    val_list = video_jsonl_prompt_list[train_end:val_end]
    test_list = video_jsonl_prompt_list[val_end:]
    return train_list, val_list, test_list


def main():
    """
    Main pipeline:
      1) Finds all (video, .jsonl) in base_dir.
      2) Splits into train/val/test by p%.
      3) Chunks each subset into subclips, writes subclips and metadata to
         train_set/, val_set/, test_set/.
      4) Also writes videos_{split}.txt, prompts_{split}.txt, images_{split}.txt
         inside train_set/, val_set/, test_set/.
    """
    #base_dir = "/data/cvg/sebastian/minecraft_basalt/14"  # Where your raw .mp4 + .jsonl live
    #data_dir = "/data/cvg/sebastian/minecraft_basalt/test_processed2"  # Where subclips + metadata go
    base_dir = "data/processed_gf"
    data_dir = "data/processed_gf_3"  # Where subclips + metadata go

    postfix = "_" + base_dir.split("/")[-1]  # e.g. "14"

    # Adjust these so they add up to 1.0
    train_split = 0.8
    val_split = 0.15
    test_split = 0.05  # set > 0 for a separate test set

    chunk_size = -1 #49  # frames per chunk
    desired_fps = None #10 #None  # e.g. set =16 if you want to re-encode at 16fps
    n_workers = 1

    # 1) Find .mp4 files
    video_paths = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)

    # 2) Match video => .jsonl => prompt
    video_jsonl_prompt_list = []
    for vp in video_paths:
        base_no_ext = os.path.splitext(vp)[0]
        jp = base_no_ext + ".jsonl"
        if os.path.exists(jp):
            subfolder_prompt = find_subfolder_prompt(vp, base_dir)
            video_jsonl_prompt_list.append((vp, jp, subfolder_prompt))
        else:
            print(f"Warning: no matching .jsonl for {vp} -- skipping.")

    # 3) Partition dataset
    train_list, val_list, test_list = partition_dataset(
        video_jsonl_prompt_list,
        train_split, val_split, test_split,
        seed=42
    )

    def process_split(split_name, subset_list, postfix):
        """
        For each subset (train, val, test):
         - create data_dir/<split_name>_set/
         - chunk videos -> .mp4 subclips
         - save metadata
         - generate videos_{split_name}.txt, prompts_{split_name}.txt, images_{split_name}.txt
           in the same folder (e.g. data_dir/train_set/).
        """
        if not subset_list:
            return  # nothing to do

        split_dir = os.path.join(data_dir, f"{split_name}_set")
        os.makedirs(split_dir, exist_ok=True)

        # Make sure the subfolders exist for chunked outputs
        os.makedirs(os.path.join(split_dir, "videos" + postfix), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "metadata" + postfix), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "first_frames" + postfix), exist_ok=True)

        all_chunked_entries = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_one_file,
                    triple,
                    split_dir,
                    chunk_size,
                    desired_fps,
                    postfix
                ): triple
                for triple in subset_list
            }
            with tqdm(total=len(futures), desc=f"Processing {split_name} files") as pbar:
                for fut in as_completed(futures):
                    try:
                        # Each future returns a list of (chunk_video_path, prompt, chunk_image_path)
                        result_list = fut.result()
                        all_chunked_entries.extend(result_list)
                    except Exception as e:
                        print(f"Error in {split_name} file processing: {e}")
                    finally:
                        pbar.update(1)

        # Create parallel text files for this split

        videos_txt_path = os.path.join(split_dir, f"videos_{split_name + postfix}.txt" )
        prompts_txt_path = os.path.join(split_dir, f"prompts_{split_name + postfix}.txt")
        images_txt_path = os.path.join(split_dir, f"images_{split_name + postfix}.txt")

        with open(videos_txt_path, 'w', encoding='utf-8') as vf, \
             open(prompts_txt_path, 'w', encoding='utf-8') as pf, \
             open(images_txt_path, 'w', encoding='utf-8') as imf:

            for (video_path, prompt, image_path) in all_chunked_entries:
                # We write relative paths so the .txt files are portable
                rel_video_path = os.path.relpath(video_path, split_dir)
                rel_image_path = os.path.relpath(image_path, split_dir)

                vf.write(f"{rel_video_path}\n")
                pf.write(f"{prompt}\n")
                imf.write(f"{rel_image_path}\n")

        print(f"Finished processing {split_name} set!")
        print(f"  -> {videos_txt_path}")
        print(f"  -> {prompts_txt_path}")
        print(f"  -> {images_txt_path}")

    # 4) Process each subset
    process_split("train", train_list, postfix)
    process_split("val", val_list, postfix)
    if test_split > 0.0:
        process_split("test", test_list, postfix)

    print("All splitting and chunking complete!")


#if __name__ == "__main__":
#
#    base_dir = "/data/cvg/sebastian/minecraft_basalt/14"
#    data_dir = "/data/cvg/sebastian/minecraft_basalt/test_processed"
#    chunk_size = 49
#    desired_fps = None
#
#    video_paths = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)
#    video_path = video_paths[0]  # Pick one video to test
#    jsonl_path = video_path.replace(".mp4", ".jsonl")
#
#    if os.path.exists(jsonl_path):
#        subfolder_prompt = find_subfolder_prompt(video_path, base_dir)
#        process_one_file((video_path, jsonl_path, subfolder_prompt), data_dir, chunk_size, desired_fps,)
#    else:
#        print("JSONL file missing for test video.")

if __name__ == "__main__":
    main()