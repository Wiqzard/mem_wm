import os
import json
import math
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

from moviepy.video.io.VideoFileClip import VideoFileClip

################################################################################
# ADAPT THESE MAPPINGS FOR YOUR 8 SUBFOLDERS AND THEIR PROMPTS
################################################################################
SUBFOLDER_PROMPTS = {
    "8": "The video depicts a scene within the Minecraft game environment where a player starts in a new world and builds a simple house. The house is constructed primarily with wood, dirt, and sand, along with crafted wooden items like doors and fences. The player avoids using advanced materials like stone. The structure is decorated with a personal touch but contains no specific furniture, capturing the charm of a cozy beginner's build.",
    "9":  "In this video, the player uses the provided resources to build a house, home, or structure in a short time frame. The Minecraft world comes alive as the player works quickly, aiming to complete the build before time runs out. The video highlights the urgency and creativity of building under constraints.",
    "10": "This video follows a player in a new Minecraft world as they attempt to craft a diamond pickaxe. The player avoids searching for villages or using glitches and instead explores caves and mines for diamonds. Bad luck with some seeds adds an element of suspense, as the player works against the odds to achieve their goal in the natural game environment.",
    "11": "The video depicts a player searching for a cave in the Minecraft world. The player explores the surface for cave entrances without digging directly down. Upon finding a cave, leaving the viewer curious about what lies ahead in the depths.",
    "12": "Set in a mountainous Minecraft biome, the video shows a player with a water bucket and tools creating a stunning waterfall. After completing the build, the player repositions themselves to capture a scenic view of the waterfall. The video concludes with the player quitting the game, preserving the picturesque moment.",
    "13": "The video takes place in a Minecraft village, where the player builds an animal pen next to one of the houses. Using fence posts, they create a pen containing at least two animals of the same type, such as chickens, cows, pigs, sheep, or rabbits. The pen includes a gate for easy access, and any accidental extra animals are removed to maintain the focus on the task. The peaceful village remains unharmed throughout the process.",
    "14": "In this video, the player spawns in a Minecraft village and uses the items in their inventory to construct a new house in the style of the village. The house is built in an appropriate location, such as next to the village path, and integrates seamlessly with the surrounding environment. After completing the build, the player provides a brief tour, slowly panning around to showcase the walls and roof, capturing the essence of village architecture.",
}
################################################################################


def filter_action_fields(action_dict):
    """
    Filters out only the fields we need from each line of the .jsonl action.
    
    From 'keyboard', keep whether keys in [w,a,s,d,space,shift] are pressed.
    From 'mouse', keep dx, dy, and 'buttons'.
    """
    relevant_keys = {
        "key.keyboard.w": "w",
        "key.keyboard.a": "a",
        "key.keyboard.s": "s",
        "key.keyboard.d": "d",
        "key.keyboard.space": "space",
        "key.keyboard.left.shift": "shift",
        "key.keyboard.right.shift": "shift"
    }

    filtered = {}
    
    # Extract relevant mouse fields if present
    mouse_info = action_dict.get("mouse", {})
    filtered["dx"] = mouse_info.get("dx", 0.0)
    filtered["dy"] = mouse_info.get("dy", 0.0)
    filtered["buttons"] = mouse_info.get("buttons", [])
    
    # Extract relevant keyboard fields
    keyboard_info = action_dict.get("keyboard", {})
    pressed_keys = keyboard_info.get("keys", [])
    # Map raw key strings (e.g., "key.keyboard.w") to simplified representation
    relevant_pressed = [relevant_keys[k] for k in pressed_keys if k in relevant_keys]
    
    filtered["keys"] = relevant_pressed  # e.g. ["w", "a"]
    
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
            try:
                action_dict = json.loads(line)
            except json.JSONDecodeError:
                # Malformed line => skip or handle
                continue
            
            filtered = filter_action_fields(action_dict)
            actions.append(filtered)
    return actions

def split_video_and_actions(
    video_path,
    jsonl_path,
    data_dir,
    chunk_size,
    desired_fps=30.0
):
    """
    Splits the video into chunks of size `chunk_size` frames, at `desired_fps`.
    Splits the .jsonl actions accordingly (assuming 1 action per frame).
    
    Returns a list of (chunk_video_path, chunk_metadata_path).
    """
    # 1) Read & filter actions
    actions = read_and_filter_actions(jsonl_path)
    total_action_frames = len(actions)
    
    # 2) Load the video with MoviePy
    clip = VideoFileClip(video_path)
    original_duration = clip.duration  # in seconds
    
    # We'll define total frames based on len(actions):
    total_frames = total_action_frames
    
    # 3) Number of chunks
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    num_chunks = math.ceil(total_frames / chunk_size)
    
    # 4) Prepare output dirs
    video_out_dir = os.path.join(data_dir, "videos")
    meta_out_dir = os.path.join(data_dir, "metadata")
    os.makedirs(video_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)

    # We'll create a base_name from the relative path of the .mp4
    # e.g.  subfolder2/myvideo => subfolder2_myvideo
    rel_video_path = os.path.relpath(video_path, start=os.path.dirname(video_path))
    base_name = os.path.splitext(rel_video_path)[0].replace(os.sep, "_")

    result_pairs = []
    
    for i in range(num_chunks):
        chunk_index = i + 1
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, total_frames)
        if start_frame >= end_frame:
            continue
        
        # Convert frames to time in the original clip
        start_time_sec = (start_frame / total_frames) * original_duration
        end_time_sec   = (end_frame   / total_frames) * original_duration
        
        subclip = clip.subclip(start_time_sec, end_time_sec)
        
        # Construct output file names
        chunk_video_name = f"{base_name}_chunk_{chunk_index}.mp4"
        chunk_json_name  = f"{base_name}_chunk_{chunk_index}.json"
        
        chunk_video_path = os.path.join(video_out_dir, chunk_video_name)
        chunk_json_path  = os.path.join(meta_out_dir, chunk_json_name)
        
        # Re-encode the subclip at desired_fps
        subclip.write_videofile(
            chunk_video_path,
            fps=desired_fps,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )
        
        # Slice out the relevant actions
        chunk_actions = actions[start_frame:end_frame]
        chunk_metadata = {"actions": chunk_actions}
        
        with open(chunk_json_path, 'w', encoding='utf-8') as jf:
            json.dump(chunk_metadata, jf, indent=2)
        
        result_pairs.append((chunk_video_path, chunk_json_path))
        
        subclip.close()
    
    clip.close()
    return result_pairs

def process_one_file(video_jsonl_prompt_tuple, data_dir, chunk_size, desired_fps):
    """
    Process a single (video, jsonl, prompt) triple.
    
    Returns a list of (chunked_video_path, prompt) so we can line up
    videos.txt and prompts.txt.
    """
    video_path, jsonl_path, subfolder_prompt = video_jsonl_prompt_tuple
    
    pairs = split_video_and_actions(
        video_path=video_path,
        jsonl_path=jsonl_path,
        data_dir=data_dir,
        chunk_size=chunk_size,
        desired_fps=desired_fps
    )
    # pairs is a list of (chunk_video_path, chunk_metadata_path)
    
    # We'll return (chunk_video_path, prompt) for each chunk
    results = [(p[0], subfolder_prompt) for p in pairs]
    return results

def find_subfolder_prompt(video_path, base_dir):
    """
    Given a video path, find which subfolder it belongs to (the first directory
    level inside base_dir), then return the matching prompt from SUBFOLDER_PROMPTS.
    
    If you have a more complicated directory structure, adapt as needed.
    """
    # relparts[0] should be the subfolder name if the structure is base_dir/subfolderX/file.mp4
    rel_parts = os.path.relpath(video_path, base_dir).split(os.sep)
    subfolder = rel_parts[0]
    
    # Retrieve the prompt
    prompt = SUBFOLDER_PROMPTS.get(subfolder, "No prompt found")  # or handle unknown subfolders
    return prompt

def main():
    """
    Main driver for chunking:
      - We have multiple subfolders under base_dir, each with its own prompt.
      - For each (video, .jsonl), we chunk them and store:
           data_dir/videos/
           data_dir/metadata/
      - We then write videos.txt and prompts.txt in parallel lines.
    """
    base_dir = "/home/ss24m050/Documents/CogVideo/data_test/pre"  # "/path/to/dataset"
    data_dir = "/home/ss24m050/Documents/CogVideo/data_test/post"
    chunk_size = 100  # frames per chunk
    desired_fps = 16 #30.0
    n_workers = 4
    
    # 1) Find all .mp4 files
    video_paths = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)
    
    # 2) Match each video to its .jsonl and a subfolder prompt
    #    If <relpath>.mp4 => <relpath>.jsonl
    video_jsonl_prompt_list = []
    for vp in video_paths:
        base_no_ext = os.path.splitext(vp)[0]
        jp = base_no_ext + ".jsonl"
        if os.path.exists(jp):
            # Identify which subfolder this belongs to
            subfolder_prompt = find_subfolder_prompt(vp, base_dir)
            video_jsonl_prompt_list.append((vp, jp, subfolder_prompt))
        else:
            # If no matching .jsonl, skip or handle accordingly
            pass
    
    # Prepare directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "metadata"), exist_ok=True)
    
    # We'll collect final chunk info as (video_path, prompt)
    all_chunked_entries = []
    
    # 3) Parallel chunking
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_one_file, triple, data_dir, chunk_size, desired_fps): triple
            for triple in video_jsonl_prompt_list
        }
        for fut in as_completed(futures):
            result_list = fut.result()  # list of (chunk_video_path, prompt)
            all_chunked_entries.extend(result_list)
    
    # 4) Write videos.txt and prompts.txt in parallel lines
    videos_txt_path = os.path.join(data_dir, "videos.txt")
    prompts_txt_path = os.path.join(data_dir, "prompts.txt")
    
    with open(videos_txt_path, 'w', encoding='utf-8') as vf, \
         open(prompts_txt_path, 'w', encoding='utf-8') as pf:
        for (video_path, prompt) in all_chunked_entries:
            rel_video_path = os.path.relpath(video_path, data_dir)
            vf.write(f"{rel_video_path}\n")
            pf.write(f"{prompt}\n")
    
    print("Chunking complete!")
    print(f"videos.txt => {videos_txt_path}")
    print(f"prompts.txt => {prompts_txt_path}")

if __name__ == "__main__":
    main()
