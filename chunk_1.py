import os
import json
import math
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# MoviePy for video handling; install if needed:
#    pip install moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip


def filter_action_fields(action_dict):
    """
    Filters out only the fields we need from each line of the .jsonl action.

    From 'keyboard', keep only whether the pressed keys include w, a, s, d, space, shift.
    From 'mouse', keep dx, dy, and 'buttons'.
    """
    # Relevant keys
    relevant_keys = {
        "key.keyboard.w": "w",
        "key.keyboard.a": "a",
        "key.keyboard.s": "s",
        "key.keyboard.d": "d",
        "key.keyboard.space": "space",
        "key.keyboard.left.shift": "shift",
        "key.keyboard.right.shift": "shift",
    }  # treat left or right shift as "shift"

    filtered = {}

    # Extract relevant mouse fields if present
    mouse_info = action_dict.get("mouse", {})
    filtered["dx"] = mouse_info.get("dx", 0.0)
    filtered["dy"] = mouse_info.get("dy", 0.0)
    filtered["buttons"] = mouse_info.get("buttons", [])

    # Extract relevant keyboard fields
    # We only note if certain keys appear in action_dict["keyboard"]["keys"]
    keyboard_info = action_dict.get("keyboard", {})
    pressed_keys = keyboard_info.get("keys", [])
    # Map raw key strings (e.g., "key.keyboard.w") to our simplified representation
    relevant_pressed = [relevant_keys[k] for k in pressed_keys if k in relevant_keys]

    filtered["keys"] = relevant_pressed  # e.g. ["w", "a"]

    return filtered


def read_and_filter_actions(jsonl_path):
    """
    Reads each line from the .jsonl path, parses it as JSON, and filters
    out only the relevant fields. Returns a list of filtered actions.
    """
    actions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                action_dict = json.loads(line)
            except json.JSONDecodeError:
                # If a line is malformed, skip or handle as needed
                continue

            filtered = filter_action_fields(action_dict)
            actions.append(filtered)
    return actions


def split_video_and_actions(
    video_path,
    jsonl_path,
    data_dir,
    chunk_size,
    desired_fps=30.0,
):
    """
    Splits the video into chunks of size `chunk_size` frames, re-encoded at `desired_fps`.
    Splits the corresponding .jsonl actions accordingly, assuming 1 action per frame.

    Parameters
    ----------
    video_path : str
        Path to the input .mp4 file.
    jsonl_path : str
        Path to the corresponding .jsonl file (one JSON action per line).
    data_dir : str
        Base output directory. Will contain `videos/` and `metadata/`.
    chunk_size : int
        Number of frames per chunk.
    desired_fps : float
        FPS to which we will re-encode the video.

    Returns
    -------
    list of (str, str)
        A list of (chunk_video_path, chunk_metadata_path) pairs.
    """
    # 1) Read and filter actions
    actions = read_and_filter_actions(jsonl_path)

    # The total "action frames" is simply len(actions)
    total_action_frames = len(actions)

    # 2) Load the video with MoviePy
    clip = VideoFileClip(video_path)

    # 3) If the original clip's FPS is not desired_fps, we will re-encode
    #    but for chunking we still want to figure out how to subclip by frames
    #    relative to the new FPS.
    #    If the clip is T seconds, and we want desired_fps, then total_frames = T * desired_fps (approx).
    original_duration = clip.duration  # in seconds
    # We'll define total_frames based on the *action file*,
    # since you indicated 1 action line per frame. If they differ widely,
    # you may need different logic. For now, we assume actions lines = actual frames.

    total_frames = total_action_frames  # from action file

    # If the action file is missing lines or has more lines than the video length,
    # you might need additional checks. We'll keep it simple here.

    # 4) Compute the number of chunks
    num_chunks = math.ceil(total_frames / chunk_size) if chunk_size > 0 else 1

    # Prepare output directories
    video_out_dir = os.path.join(data_dir, "videos")
    meta_out_dir = os.path.join(data_dir, "metadata")
    os.makedirs(video_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.relpath(video_path, start=os.path.dirname(video_path)))[0]
    # For example, if relpath is "v7.3/contractor1-abc-2022-12-12-10.14", base_name might be that entire path
    # but typically you'd want just the last part, or keep it for clarity.
    # You can adapt as needed.

    result_pairs = []

    for i in range(num_chunks):
        chunk_index = i + 1
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, total_frames)  # non-inclusive end

        if start_frame >= end_frame:
            continue

        # Convert frames to time in the original clip. This is approximate,
        # because we assume the total frames match the length. We do:
        #   fraction_of_total = (frame_number / total_frames)
        #   time_in_seconds   = fraction_of_total * original_duration
        # A more direct approach would be to consider the new FPS.
        # If total_frames = actions, we want each line to correspond to 1 "frame" of time.
        # Then each "frame" in the chunk is 1 / desired_fps seconds.

        start_time_sec = (start_frame / total_frames) * original_duration
        end_time_sec = (end_frame / total_frames) * original_duration

        # Subclip at [start_time_sec, end_time_sec)
        subclip = clip.subclip(start_time_sec, end_time_sec)

        # Re-encode the subclip at desired_fps
        chunk_video_name = f"{base_name}_chunk_{chunk_index}.mp4"
        chunk_json_name = f"{base_name}_chunk_{chunk_index}.json"

        chunk_video_path = os.path.join(video_out_dir, chunk_video_name)
        chunk_json_path = os.path.join(meta_out_dir, chunk_json_name)

        # Write the chunked video, forcing a new fps if desired
        subclip.write_videofile(
            chunk_video_path, fps=desired_fps, codec="libx264", audio_codec="aac", logger=None
        )

        # Build the chunked action list
        chunk_actions = actions[start_frame:end_frame]

        # You can store them however you like. For consistency, let's store them as:
        # {
        #   "actions": [ {dx, dy, buttons, keys}, {}, ... ]
        # }
        chunk_metadata = {"actions": chunk_actions}

        # Write the chunked metadata
        with open(chunk_json_path, "w", encoding="utf-8") as jf:
            json.dump(chunk_metadata, jf, indent=2)

        result_pairs.append((chunk_video_path, chunk_json_path))

        subclip.close()

    clip.close()
    return result_pairs


def process_one_file(video_jsonl_pair, data_dir, chunk_size, desired_fps):
    """
    Process a single (video, jsonl) pair. Returns a list of chunked video paths
    for later logging into videos.txt.
    """
    video_path, jsonl_path = video_jsonl_pair
    chunk_paths = split_video_and_actions(
        video_path=video_path,
        jsonl_path=jsonl_path,
        data_dir=data_dir,
        chunk_size=chunk_size,
        desired_fps=desired_fps,
    )
    # chunk_paths is a list of (chunk_video_path, chunk_metadata_path)
    return [cp[0] for cp in chunk_paths]


def main():
    """
    Main driver for the chunking script:
     - Gathers .mp4 and .jsonl pairs from a base directory
     - Splits them in parallel
     - Generates videos.txt listing chunked videos
    """
    base_dir = "/home/ss24m050/Documents/CogVideo/data_test/pre"  # "/path/to/dataset"
    data_dir = "/home/ss24m050/Documents/CogVideo/data_test/post"
    chunk_size = 100  # frames per chunk
    desired_fps = 16 #30.0
    n_workers = 4

    # 1) Find all .mp4 files under base_dir
    video_paths = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)

    # 2) Match each video with its .jsonl file
    #    If <relpath>.mp4 => <relpath>.jsonl
    video_jsonl_pairs = []
    for vp in video_paths:
        rel = os.path.splitext(vp)[0]  # remove .mp4 => /path/to/.../v7.3/session-...
        jp = rel + ".jsonl"
        if os.path.exists(jp):
            video_jsonl_pairs.append((vp, jp))
        else:
            # No matching JSONL => skip or warn
            pass

    # 3) Ensure output directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "metadata"), exist_ok=True)

    # 4) Parallel processing
    all_chunked_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_pair = {
            executor.submit(process_one_file, pair, data_dir, chunk_size, desired_fps): pair
            for pair in video_jsonl_pairs
        }
        for future in as_completed(future_to_pair):
            result_video_paths = future.result()
            all_chunked_paths.extend(result_video_paths)

    # 5) Write videos.txt
    videos_txt_path = os.path.join(data_dir, "videos.txt")
    with open(videos_txt_path, "w", encoding="utf-8") as vf:
        for p in all_chunked_paths:
            rel_path = os.path.relpath(p, data_dir)
            vf.write(f"{rel_path}\n")

    print(f"Chunking complete. Videos and metadata stored in: {data_dir}")
    print(f"Video list written to: {videos_txt_path}")


if __name__ == "__main__":
    main()
