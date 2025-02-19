import os
import shutil

# Path to the main directory
root_dir = '/capstor/scratch/cscs/sstapf/mem_wm/finetune/wandb'

# Iterate over each subfolder in the root directory
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    
    # Ensure that we're dealing with directories or symlinks (that may point to directories)
    if os.path.isdir(subfolder_path) or os.path.islink(subfolder_path):
        # Define the path to the 'files' folder within the subfolder
        files_dir = os.path.join(subfolder_path, 'files')
        
        if os.path.isdir(files_dir):
            # Define the path to the 'media' folder within 'files'
            media_dir = os.path.join(files_dir, 'media')
            if not os.path.isdir(media_dir):
                print(f"Deleting '{subfolder_path}' because 'media' subfolder is missing in 'files'.")
                if os.path.islink(subfolder_path):
                    os.unlink(subfolder_path)
                else:
                    shutil.rmtree(subfolder_path)
        else:
            # Optionally, if the 'files' folder doesn't exist, delete the folder as well.
            print(f"Deleting '{subfolder_path}' because 'files' folder is missing.")
            if os.path.islink(subfolder_path):
                os.unlink(subfolder_path)
            else:
                shutil.rmtree(subfolder_path)
