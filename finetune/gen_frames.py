import os
import imageio.v3 as iio
import multiprocessing
from pathlib import Path
from tqdm import tqdm

def extract_first_frame(video_path, output_dir, rel_paths):
    try:
        # Read the first frame
        first_frame = iio.imread(video_path, index=0)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output file name with original format
        output_path = output_dir / (video_path.stem + video_path.suffix.replace('.mp4', '.png'))
        
        # Save the first frame as an image
        iio.imwrite(output_path, first_frame)
        
        # Store relative path
        rel_paths.append(str(output_path.relative_to(output_dir.parent)))
    except Exception as e:
        # In a multiprocessing context, it's often helpful to re-raise the exception
        # so the main process can handle/log it. Alternatively, you can just pass.
        raise RuntimeError(f"Failed to process {video_path}: {e}")

if __name__ == "__main__":
    base_folder = Path("/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/train_set")
    video_dir = base_folder / "videos"
    output_dir = base_folder / "first_frames"
    images_txt_path = base_folder / "images.txt"
    
    # Get list of all video files
    video_files = list(video_dir.glob("*.mp4"))  # Modify as needed to support other formats
    
    # Prepare a multiprocessing manager list for storing paths
    manager = multiprocessing.Manager()
    rel_paths = manager.list()
    
    # Decide on number of workers
    num_workers = min(multiprocessing.cpu_count(), len(video_files))
    
    # Create a Pool
    with multiprocessing.Pool(num_workers) as pool:
        results = []
        # Submit tasks asynchronously
        for video in video_files:
            results.append(
                pool.apply_async(extract_first_frame, args=(video, output_dir, rel_paths))
            )
        
        # Use tqdm to show progress while we collect results
        for _ in tqdm(results, desc="Processing videos", total=len(video_files)):
            # .get() will raise an exception if one occurred in the worker
            _.get()
    
    # Write relative paths to images.txt
    with open(images_txt_path, "w") as f:
        f.write("\n".join(rel_paths))
    
    print(f"Relative paths saved to {images_txt_path}")