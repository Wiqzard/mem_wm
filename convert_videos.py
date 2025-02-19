import os
import glob
import multiprocessing as mp
from moviepy.editor import VideoFileClip
import cv2
import numpy as np

def process_video(video_path):
    """
    Reads a video using MoviePy, re-encodes it correctly, and overwrites the original file.
    """
    try:
        clip = VideoFileClip(video_path)
        
        # Ensure full duration is processed
        clip = clip.subclip(0, clip.duration)
        
        # Convert frames to avoid color inversion issue
        #def fix_colors(frame):
        #    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #fixed_clip = clip.fl_image(fix_colors)
        fixed_clip = clip
        
        temp_path = video_path + ".tmp.mp4"
        fixed_clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", fps=clip.fps)
        clip.close()
        
        # Overwrite the original file
        os.replace(temp_path, video_path)
        print(f"Re-encoded: {video_path}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def main(folder_path):
    """
    Finds all MP4 videos in the folder and processes them in parallel.
    """
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    if not video_files:
        print("No MP4 videos found in the folder.")
        return

    # Use multiprocessing to speed up processing
    num_workers = min(mp.cpu_count(), len(video_files))
    with mp.Pool(num_workers) as pool:
        pool.map(process_video, video_files)

if __name__ == "__main__":
    #folder_path = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed/train_set/videos_gf_processed"
    folder_path = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed/test_set/videos_gf_processed"
    #folder_path = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed/val_set/videos_gf_processed"

    #folder_path = "/home/ss24m050/Documents/CogVideo/data/processed_gf_3/train_set/videos_processed_gf"  # Change this to your actual folder path
    main(folder_path)