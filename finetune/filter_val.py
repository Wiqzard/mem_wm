##!/usr/bin/env python3
#input_file = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set/images.txt"  # Replace with your actual input file name
#output_file = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set/images_new.txt"  # Replace with your desired output file name
#
#with open(input_file, "r") as infile, open(output_file, "w") as outfile:
#    for line in infile:
#        modified_line = "/".join(line.strip().split("/")[1:])  # Remove the first folder
#        outfile.write(modified_line + "\n")
#
#print(f"Modified paths saved to {output_file}")
#

import os

# Paths (adjust if needed)
base_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set"
images_txt = os.path.join(base_dir, "images_new.txt")
filtered_txt = os.path.join(base_dir, "images_filtered.txt")

# The directory where the images should be found
first_frames_dir = os.path.join(base_dir, "first_frames")

def main():
    # Read all lines from images.txt
    with open(images_txt, "r") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    valid_lines = []

    for line in lines:
        # If lines in images.txt are relative filenames, construct the full path:
        # e.g., line might be "12345.jpg", so we do ...
        image_path = os.path.join(first_frames_dir, line)
        print(image_path)

        # Check if this image path exists
        if os.path.isfile(image_path):
            valid_lines.append(line)

    # Write the filtered lines to images_filtered.txt
    with open(filtered_txt, "w") as f_out:
        for valid_line in valid_lines:
            f_out.write(valid_line + "\n")

    print(f"Filtering complete. Valid entries: {len(valid_lines)}")
    print(f"Filtered list written to: {filtered_txt}")

if __name__ == "__main__":
    main()
