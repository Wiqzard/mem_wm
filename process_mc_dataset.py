import os
import csv
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_clips_and_prompts(
    source_root,
    output_root,
    dataset_dirs = ("data_2003", "data_269"),
    video_subdir = "video",
    annotation_filename = "annotation.csv",
    frame_buffer = 10
):
    """
    Extracts sub-clips based on 'annotation.csv' files in the specified dataset directories
    (e.g., data_2003 and data_269), then saves them along with prompt text in the requested format.
    
    :param source_root: Path to the GF-Minecraft/ folder containing data_2003, data_269, etc.
    :param output_root: Path to where the new structure (prompts.txt, videos/, videos.txt) will be created.
    :param dataset_dirs: A tuple/list of sub-folder names to look for annotation.csv in (default: data_2003, data_269).
    :param video_subdir: Name of the folder within each dataset_dir that contains .mp4 files.
    :param annotation_filename: Name of the CSV file containing annotation data.
    :param frame_buffer: Number of additional frames to add at the end of each clip.
    """
    
    # Create output directories if they don't exist
    os.makedirs(output_root, exist_ok=True)
    videos_out_dir = os.path.join(output_root, "videos")
    os.makedirs(videos_out_dir, exist_ok=True)
    
    # Prepare output text files
    prompts_path = os.path.join(output_root, "prompts.txt")
    videos_txt_path = os.path.join(output_root, "videos.txt")
    
    with open(prompts_path, "w", encoding="utf-8") as prompts_file, \
         open(videos_txt_path, "w", encoding="utf-8") as videos_file:
        
        clip_count = 0
        
        # Iterate over dataset folders (e.g., data_2003, data_269)
        for dset in dataset_dirs:
            dset_path = os.path.join(source_root, dset)
            annotation_csv_path = os.path.join(dset_path, annotation_filename)
            
            # Skip if annotation.csv doesn't exist
            if not os.path.isfile(annotation_csv_path):
                print(f"Warning: annotation.csv not found in {dset_path}, skipping.")
                continue
            
            # Read annotation.csv
            with open(annotation_csv_path, "r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                
                for row in reader:
                    # Example columns in annotation.csv:
                    # "Original video name", "Start frame index", "End frame index", "Prompt"
                    original_video_name = row["original video name"]
                    start_frame = int(row["start frame index"])
                    end_frame = int(row["end frame index"])
                    prompt_text = row["prompt"].strip()
                    
                    # Construct path to the original video
                    video_path = os.path.join(dset_path, video_subdir, original_video_name)
                    
                    if not os.path.isfile(video_path):
                        print(f"Video file not found: {video_path}. Skipping.")
                        continue
                    
                    # Extract sub-clip using MoviePy
                    try:
                        with VideoFileClip(video_path) as video_clip:
                            fps = video_clip.fps
                            
                            start_time = start_frame / fps
                            end_time = (end_frame + frame_buffer) / fps
                            
                            # Ensure end_time doesn't exceed the video duration
                            end_time = min(end_time, video_clip.duration)
                            
                            sub_clip = video_clip.subclip(start_time, end_time)
                            
                            # Unique name for the output sub-clip
                            clip_count += 1
                            output_clip_name = f"clip_{clip_count:05d}.mp4"
                            output_clip_path = os.path.join(videos_out_dir, output_clip_name)
                            
                            # Write the sub-clip to disk
                            # Using 'codec="libx264", audio_codec="aac"' is typical for MP4
                            sub_clip.write_videofile(
                                output_clip_path,
                                codec="libx264",
                                audio_codec="aac",
                                # you can adjust the bitrate or other parameters as needed
                                verbose=False,
                                logger=None  # avoid spam in console
                            )
                            
                            # Write prompt and sub-clip filename to our output text files
                            prompts_file.write(prompt_text + "\n")
                            videos_file.write(output_clip_name + "\n")
                    
                    except Exception as e:
                        print(f"Error processing {video_path} [{start_frame}-{end_frame}]: {e}")
                        continue


if __name__ == "__main__":
    # Example usage:
    source_dir = "/data/cvg/sebastian/minecraft_gf/GameFactory-Dataset/GF-Minecraft" 
    output_dir = "data/mc" # os.path.join(source_dir, "processed")  # "GF-Minecraft/processed"
    
    extract_clips_and_prompts(
        source_root=source_dir,
        output_root=output_dir,
        dataset_dirs=("data_269", ),  # or just ["data_2003", "data_269"]
        video_subdir="video",
        annotation_filename="annotation.csv",
        frame_buffer=10
    )
    
    print(f"Done! Processed dataset is saved under: {output_dir}")
