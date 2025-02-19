#!/usr/bin/env python3

import os
import csv
import json
import shutil

# Root folder containing GF-Minecraft/data_XXX subfolders
ROOT_DIR = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/GF-Minecraft/data_2003" #/data_2003" #"GF-Minecraft"
OUTPUT_DIR = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/gf_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_actions_to_new_json(actions_dict):
    """
    Convert the 'actions' dictionary from the original format into
    a list of line-based JSON records in the new format.
    """
    output_lines = []
    # We ignore the "0" frame, per your dataset specs
    valid_keys = sorted(k for k in actions_dict.keys() if k.isdigit() and k != "0")
    
    for frame_str in valid_keys:
        frame_idx = int(frame_str)
        
        entry = actions_dict[frame_str]
        # entry typically has: ws, ad, scs, pitch_delta, yaw_delta, etc.
        
        # Convert movement codes to key strings
        keys_held = []
        
        ws = entry.get("ws", 0)
        if ws == 1:
            keys_held.append("key.keyboard.w")
        elif ws == 2:
            keys_held.append("key.keyboard.s")

        ad = entry.get("ad", 0)
        if ad == 1:
            keys_held.append("key.keyboard.a")
        elif ad == 2:
            keys_held.append("key.keyboard.d")

        scs = entry.get("scs", 0)
        if scs == 1:
            keys_held.append("key.keyboard.space")
        elif scs == 2:
            keys_held.append("key.keyboard.shift")
        elif scs == 3:
            keys_held.append("key.keyboard.ctrl")

        # Multiply pitch_delta, yaw_delta by 15 for degrees
        pitch_delta = entry.get("pitch_delta", 0.0) * 15
        yaw_delta   = entry.get("yaw_delta", 0.0)   * 15
        
        line_dict = {
            "mouse": {
                "x": 0.0,       
                "y": 0.0,       
                "dx": yaw_delta,   
                "dy": pitch_delta, 
                "scaledX": 0.0,    
                "scaledY": 0.0,
                "dwheel": 0.0,
                "buttons": [],
                "newButtons": []
            },
            "keyboard": {
                "keys": keys_held,
                "newKeys": [],
                "chars": ""
            },
            "hotbar": 0,
            "tick": frame_idx,    # or frame_idx - 1, up to you
            "isGuiOpen": False
        }
        
        output_lines.append(line_dict)
    return output_lines


def process_data_folder(data_folder):
    """
    Given a path like GF-Minecraft/data_269, this will:
    1. Look in metadata/ for any .json files.
    2. Convert each .json using the convert_actions_to_new_json function.
    3. Copy the corresponding .mp4 from video/ if it exists.
    4. Write out the new line-based JSON to OUTPUT_DIR.
    """
    metadata_dir = os.path.join(data_folder, "metadata")
    video_dir = os.path.join(data_folder, "video")
    
    # If either metadata/ or video/ is missing, skip
    if not os.path.isdir(metadata_dir) or not os.path.isdir(video_dir):
        print(f"  [!] Missing metadata/ or video/ in {data_folder}, skipping.")
        return
    
    # Iterate over all json files in metadata/
    for filename in os.listdir(metadata_dir):
        if not filename.endswith(".json"):
            continue
        
        base_name = os.path.splitext(filename)[0]  # e.g. "seed_1_part_1"
        json_path = os.path.join(metadata_dir, filename)
        
        # Corresponding video .mp4 path
        mp4_path = os.path.join(video_dir, base_name + ".mp4")
        
        if not os.path.isfile(mp4_path):
            print(f"  [!] No matching video for {json_path}, skipping.")
            continue
        
        # Read the original JSON
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        
        # Extract the actions dict
        actions_dict = data.get("actions", {})
        
        # Convert to new line-based JSON format
        converted_lines = convert_actions_to_new_json(actions_dict)
        
        # Write them out
        ##out_json_filename = base_name + ".jsonl"
        out_json_filename = base_name + ".jsonl"
        out_json_path = os.path.join(OUTPUT_DIR, out_json_filename)
        
        with open(out_json_path, "w", encoding="utf-8") as outf:
            for line_dict in converted_lines:
                json_str = json.dumps(line_dict)
                outf.write(json_str + "\n")
        
        # Copy the mp4 to the output folder
        out_mp4_filename = base_name + ".mp4"
        out_mp4_path = os.path.join(OUTPUT_DIR, out_mp4_filename)
        shutil.copyfile(mp4_path, out_mp4_path)
        
        print(f"  [+] Wrote {out_json_filename} and {out_mp4_filename}")


def main():
    """
    Walk over GF-Minecraft folder, find subfolders like data_xxx,
    process each by scanning metadata/ for JSON files and copying
    the matching MP4s from video/.
    """
    for item in os.listdir(ROOT_DIR):
        data_path = os.path.join(ROOT_DIR, item)
        if os.path.isdir(data_path) and item.startswith("data_"):
            print(f"[*] Processing folder: {data_path}")
            process_data_folder(data_path)


if __name__ == "__main__":
    main()