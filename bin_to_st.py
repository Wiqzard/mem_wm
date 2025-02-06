import os
import torch
import json
from safetensors.torch import save_file

def convert_bin_to_safetensors(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    bin_files = [f for f in os.listdir(input_dir) if f.endswith(".bin")]
    bin_files.sort()  # Ensure the order is maintained

    weight_map = {"metadata": {"total_size": 0}, "weight_map": {}}
    total_size = 0

    for i, bin_file in enumerate(bin_files, start=1):
        bin_path = os.path.join(input_dir, bin_file)
        state_dict = torch.load(bin_path, map_location="cpu")  # Load as CPU tensors

        safetensors_filename = f"diffusion_pytorch_model-{i:05d}-of-{len(bin_files):05d}.safetensors"
        safetensors_path = os.path.join(output_dir, safetensors_filename)

        # Save the weights in safetensors format
        save_file(state_dict, safetensors_path)

        # Update the weight map
        for key in state_dict.keys():
            weight_map["weight_map"][key] = safetensors_filename
            total_size += state_dict[key].numel() * state_dict[key].element_size()  # Compute total size

    weight_map["metadata"]["total_size"] = total_size

    # Save the mapping file
    with open(os.path.join(output_dir, "weights_map.json"), "w") as f:
        json.dump(weight_map, f, indent=2)

    print(f"Conversion complete! Safetensors saved in: {output_dir}")
    print(f"Weight map saved as: {os.path.join(output_dir, 'weights_map.json')}")

# Example usage
input_directory = "path/to/bin_files"  # Change this to your directory
output_directory = "path/to/output_safetensors"
convert_bin_to_safetensors(input_directory, output_directory)