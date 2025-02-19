import decord
import os
import sys
import shutil
from pathlib import Path
import random
import imageio
from PIL import Image
import json

import torch
import cv2
import numpy as np

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
from diffusers.utils import export_to_video

from finetune.models.transformer import CogVideoXTransformer3DActionModel
from finetune.models.pipeline import CogVideoXImageToVideoPipeline
from finetune.datasets.utils import load_actions_as_tensors
from finetune.utils.action_utils import generate_action_sequence



class VisualizeResults:
    def __init__(
        self,
        model_path: str,
        # Video-based inputs
        video_dir: str = None,
        video_list_file: str = None,

        # Frame+metadata-based inputs
        frames_path: str = None,
        frames_list: list = None,
        metadata_path: str = None,
        metadata_list: list = None,

        transformer_local_path: str = None,
        num_frames: int = 49,
        height: int = 352,
        width: int = 640,
        num_videos_to_generate: int = 1,
        output_dir: str = "samples",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        distributed: bool = False,
    ):
        """
        Args:
            model_path: Base path to a pretrained CogVideoX model (e.g. "THUDM/CogVideoX-2b").

            video_dir: Directory containing .mp4 videos (if provided).
            video_list_file: Text file with one .mp4 path per line (if provided).

            frames_path: Single path to one image file (e.g. a single PNG for the first frame),
                         or a directory containing multiple frames. (Used when you don't have a video.)
            frames_list: Explicit list of frame paths (e.g. ["frame_0001.png", "frame_0002.png", ...]).
            
            metadata_path: Single JSON file containing the actions/metadata for all frames.
            metadata_list: List of JSON files if you have multiple sets of actions to run in parallel
                           (mirroring frames_list).

            transformer_local_path: Optional local path for the transformer weights.
                                    If not None, overrides model_path for the transformer only.
            num_frames: Number of frames to generate per video.
            height: Output video height.
            width: Output video width.
            num_videos_to_generate: How many videos to process.
            output_dir: Where to save results.
            device: Torch device to use (usually "cuda").
            dtype: Torch dtype (e.g. torch.bfloat16).
            distributed: If True, initialize distributed (multi-GPU) generation using torch.distributed (torchrun).
        """
        self.model_path = model_path
        self.video_dir = video_dir
        self.video_list_file = video_list_file

        self.frames_path = frames_path
        self.frames_list = frames_list
        self.metadata_path = metadata_path
        self.metadata_list = metadata_list

        self.transformer_local_path = transformer_local_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_videos_to_generate = num_videos_to_generate
        self.output_dir = output_dir
        self.device = device
        self.dtype = dtype
        self.distributed = distributed

        # If distributed is True, initialize the process group.
        if self.distributed:
            import torch.distributed as dist
            dist.init_process_group(backend="nccl")
            # local_rank is set by torchrun automatically.
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.local_rank = 0  # Non-distributed scenario uses default GPU.

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Gather the inputs
        self.video_paths = []
        self.frames_and_metadata = []
        self._collect_inputs()

        # 2. Set up the pipeline
        self._setup_pipeline()
        print(f"[Rank {self.local_rank}] Pipeline using device: {self.device}")

    def _collect_inputs(self):
        """
        Collect and store the relevant inputs depending on the mode:
            - If video_dir / video_list_file is provided, we store .mp4 paths in self.video_paths.
            - If frames_path / frames_list is provided, we store them with metadata in self.frames_and_metadata.
        """
        # If we have video_dir or video_list_file:
        if (self.video_dir or self.video_list_file):
            # Collect video paths
            self.video_paths = self._collect_video_paths()

            # If frames_path/frames_list is also set, that’s conflicting usage, so we can raise an error
            if (self.frames_path or self.frames_list):
                raise ValueError(
                    "You have provided both video_dir/video_list_file and frames_path/frames_list. "
                    "Please use one mode at a time."
                )
            return

        # Otherwise, we assume frames + metadata mode:
        if (self.frames_path is None and self.frames_list is None):
            raise ValueError(
                "No video_dir/video_list_file provided, and no frames_path/frames_list provided. "
                "Cannot proceed!"
            )

        # We want to pair frames with metadata. The simplest approach is:
        # - If frames_path is not None, we assume it's a single frames path (or an image).
        # - If frames_list is not None, we assume it’s multiple items, and we must have matching metadata_list.

        if self.frames_path and self.metadata_path:
            # Single frames path + single metadata
            self.frames_and_metadata.append((self.frames_path, self.metadata_path))
        elif self.frames_list and self.metadata_list:
            if len(self.frames_list) != len(self.metadata_list):
                raise ValueError(
                    "frames_list and metadata_list must have the same length!"
                )
            for f_path, m_path in zip(self.frames_list, self.metadata_list):
                self.frames_and_metadata.append((f_path, m_path))
        else:
            raise ValueError(
                "You provided frames_path but not metadata_path (or frames_list but not metadata_list). "
                "Both must be provided for frames-based generation."
            )

    def _collect_video_paths(self):
        """Collect video paths from either a directory or a text file."""
        paths = []
        if self.video_dir and os.path.isdir(self.video_dir):
            for f in os.listdir(self.video_dir):
                if f.endswith(".mp4"):
                    paths.append(os.path.join(self.video_dir, f))
        elif self.video_list_file and os.path.isfile(self.video_list_file):
            with open(self.video_list_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        paths.append(line)
        else:
            raise ValueError(
                "No valid video_dir or video_list_file provided! "
                "Please specify either a directory with .mp4 files or a text file with one .mp4 path per line."
            )

        # Sort them so that each GPU processes a predictable subset
        paths.sort()
        return paths

    def _setup_pipeline(self):
        """Instantiate the pipeline (transformer, vae, scheduler)."""
        # 1) Transformer
        if self.transformer_local_path is not None:
            print(f"[Rank {self.local_rank}] Loading transformer from local path: {self.transformer_local_path}")
            self.transformer = CogVideoXTransformer3DActionModel.from_pretrained(
                self.transformer_local_path
            ).to(dtype=self.dtype, device=self.device)
        else:
            print(f"[Rank {self.local_rank}] Loading transformer from model_path: {self.model_path}")
            self.transformer = CogVideoXTransformer3DActionModel.from_pretrained(
                self.model_path,
                subfolder="transformer",
            ).to(dtype=self.dtype, device=self.device)

        # 2) VAE
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            self.model_path,
            subfolder="vae",
        ).to(dtype=self.dtype, device=self.device)

        # 3) Scheduler
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(
            self.model_path,
            subfolder="scheduler",
        )

        # 4) Build pipeline
        self.pipe = CogVideoXImageToVideoPipeline(
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
        ).to(self.device)

    def _load_image_as_tensor(self, image_path: str) -> torch.Tensor:
        """Helper to load a single image (PNG, JPEG, etc.) as a 4D Tensor for the pipeline."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return image

    def _load_actions(self, action_path: str):
        """
        Load the actions from a JSON path. This uses your existing 
        `load_actions_as_tensors(action_path, num_actions=...)`.
        """
        action_path = Path(action_path)
        if not action_path.is_file():
            raise FileNotFoundError(f"Action metadata not found at {action_path}")

        actions = load_actions_as_tensors(
            action_path,
            num_actions=self.num_frames-1,
        )
        return actions

    def _load_first_frame_and_actions_for_video(self, video_path: str) -> (torch.Tensor, dict):
        """
        Loads the first frame from e.g. /path/to/first_frames/<video_name>.png
        and loads associated metadata from /path/to/metadata/<video_name>.json
        """
        # 1) Build the path for the first frame
        image_path = video_path.replace("videos", "first_frames").replace(".mp4", ".png")
        # 2) Build the path for the metadata
        action_path = video_path.replace("videos", "metadata").replace(".mp4", ".json")

        image = self._load_image_as_tensor(image_path)
        actions = self._load_actions(action_path)
        return image, actions

    def _generate_video(self, image: torch.Tensor, actions: dict, seed: int = 42):
        """Generate frames using the pipeline."""
        video_result = self.pipe(
            image=image.to(self.device),
            actions=actions,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=self.num_frames,
            width=self.width,
            height=self.height,
            guidance_scale=0.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            max_sequence_length=226,
            return_dict=False,
        )
        
        generated_frames = video_result.frames[0]  # List of PIL images
        return generated_frames

    def _create_side_by_side_video(self, gt_video_path: str, generated_frames: list, output_path: str, fps: float = None):
        """
        Create a side-by-side video (GT on the left, Generated on the right)
        using `imageio` to read frames from the ground-truth video,
        and `export_to_video` from diffusers to write the result.

        If you do not have a GT video, you simply won't call this function.
        """
        print(f"[Rank {self.local_rank}] Creating side-by-side video at: {output_path}")

        # Open the ground-truth video with imageio
        vid_reader = imageio.get_reader(gt_video_path, "ffmpeg")
        meta = vid_reader.get_meta_data()

        # Attempt to retrieve FPS from metadata if not provided
        if fps is None:
            fps = meta.get("fps", 16)  # fallback to 16 if 'fps' is not available
        print(f"[Rank {self.local_rank}] Using FPS = {fps:.2f}")

        side_by_side_frames = []
        max_frames = min(self.num_frames, len(generated_frames))

        for i, frame_np in enumerate(vid_reader):
            if i >= max_frames:
                break
            gt_frame_pil = Image.fromarray(frame_np)  # from np to PIL (RGB)
            gen_frame_pil = generated_frames[i]

            # Combine them side by side
            total_width = gt_frame_pil.width + gen_frame_pil.width
            max_height = max(gt_frame_pil.height, gen_frame_pil.height)

            side_by_side = Image.new("RGB", (total_width, max_height))
            side_by_side.paste(gt_frame_pil, (0, 0))
            side_by_side.paste(gen_frame_pil, (gt_frame_pil.width, 0))

            side_by_side_frames.append(side_by_side)

        # Close reader
        vid_reader.close()

        # Save side-by-side
        export_to_video(side_by_side_frames, output_path, fps=fps)
        print(f"[Rank {self.local_rank}] Side-by-side video saved to: {output_path}")

    def run(self, seed: int = 42):
        """
        Main entry.
        - In distributed mode, splits the list of videos among different ranks (if using videos).
        - Or processes each (frames_path, metadata_path) pair if using the frames-based mode.
        """
        if self.distributed:
            import torch.distributed as dist
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # CASE 1: Video-based generation
        if False: #len(self.video_paths) > 0:
            if self.distributed:
                # Simple chunking approach for distributing videos among ranks
                chunk_size = (len(self.video_paths) + world_size - 1) // world_size
                start_idx = rank * chunk_size
                end_idx = min(start_idx + chunk_size, len(self.video_paths))
                local_video_paths = self.video_paths[start_idx:end_idx]
            else:
                local_video_paths = self.video_paths

            # Also limit to num_videos_to_generate
            random.shuffle(local_video_paths)
            local_video_paths = local_video_paths[: self.num_videos_to_generate]

            for i, video_path in enumerate(local_video_paths):
                print(f"[Rank {self.local_rank}] Processing video {i+1}/{len(local_video_paths)}: {video_path}")

                # 1) Load the first frame + actions
                try:
                    image, actions = self._load_first_frame_and_actions_for_video(video_path)
                except Exception as e:
                    print(f"[Rank {self.local_rank}] Error: {e}, skipping {video_path}")
                    continue

                # 2) Generate new frames
                generated_frames = self._generate_video(image, actions, seed=seed)

                # 3) Make side-by-side + also store just the generated
                video_name = os.path.basename(video_path)
                # side-by-side
                out_name = f"side_by_side_{os.path.splitext(video_name)[0]}.mp4"
                output_path = os.path.join(self.output_dir, out_name)
                self._create_side_by_side_video(video_path, generated_frames, output_path)

                # just the generated
                gen_name = f"gen_{os.path.splitext(video_name)[0]}.mp4"
                gen_video_path = os.path.join(self.output_dir, gen_name)
                export_to_video(generated_frames, gen_video_path)
                print(f"[Rank {self.local_rank}] Saved side-by-side and generated outputs to: {output_path}, {gen_video_path}")

            if self.distributed:
                # Sync all processes before exiting
                import torch.distributed as dist
                dist.barrier()

        # CASE 2: Frames+metadata-based generation
        else:
            # We have self.frames_and_metadata as a list of (frames_path, metadata_path) pairs
            if self.distributed:
                chunk_size = (len(self.frames_and_metadata) + world_size - 1) // world_size
                start_idx = rank * chunk_size
                end_idx = min(start_idx + chunk_size, len(self.frames_and_metadata))
                local_pairs = self.frames_and_metadata[start_idx:end_idx]
            else:
                local_pairs = self.frames_and_metadata

            # Also limit to num_videos_to_generate
            random.shuffle(local_pairs)
            local_pairs = local_pairs[: self.num_videos_to_generate]

            for idx, (f_path, m_path) in enumerate(local_pairs):
                print(f"[Rank {self.local_rank}] Processing frames+metadata {idx+1}/{len(local_pairs)}")
                print(f"  Frames Path: {f_path}")
                print(f"  Metadata Path: {m_path}")

                try:
                    # 1) Load the first frame or a single image
                    first_frame = self._load_image_as_tensor(f_path)
                    # 2) Load actions
                    if isinstance(m_path, dict):
                        actions = m_path["actions"]
                    else:
                        actions = self._load_actions(m_path)
                    # 3) Generate frames
                    print(actions)
                    generated_frames = self._generate_video(first_frame, actions, seed=seed)
                except Exception as e:
                    print(f"[Rank {self.local_rank}] Error: {e}, skipping {f_path}")
                    continue

                # 4) Save the generated frames as a .mp4
                # There's no "side-by-side" because we have no GT video
                base_name = f"frames_{idx}"
                if os.path.isdir(f_path):
                    base_name = os.path.basename(os.path.normpath(f_path))
                elif os.path.isfile(f_path):
                    base_name = os.path.splitext(os.path.basename(f_path))[0]

                gen_name = f"gen_{base_name}.mp4"
                gen_video_path = os.path.join(self.output_dir, gen_name)
                export_to_video(generated_frames, gen_video_path)
                print(f"[Rank {self.local_rank}] Saved generated video to: {gen_video_path}")

            if self.distributed:
                import torch.distributed as dist
                dist.barrier()


if __name__ == "__main__":


    seq_lr = [{"actions" : generate_action_sequence(48, "left-right")}]
    model_path = "THUDM/CogVideoX-2b" #"THUDM/CogVideoX1.5-5B-I2V" #"THUDM/CogVideoX-2b"  # For example
    transformer_local_path = "/capstor/scratch/cscs/sstapf/mem_wm/finetune/outputs/basalt_10_fps_actions_full_res/checkpoint-5800" #"/capstor/scratch/cscs/sstapf/mem_wm/finetune/outputs/outputs_2_hlr_49_cont/checkpoint-2000" # If you have a custom local path for the transformer, set it here"/capstor/scratch/cscs/sstapf/mem_wm/finetune/outputs/outputs_1.5_hlr_cont/checkpoint-3600" #
    video_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed2/test_set/videos_8" # None           #video_dir = "/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/validation_set/videos"  # Path to a directory containing .mp4 files
    video_list_file = None
    frames_path = None #"/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed2/test_set/images_test_combined.txt" #None #"/capstor/scratch/cscs/sstapf/mem_wm/data/data_269/first_frames/seed_451_part_451.png" #None         # E.g. "/path/to/single_frame.png"
    frames_list = None         # E.g. ["frame1.png", "frame2.png"]
    metadata_path = seq_lr #None       # E.g. "/path/to/actions.json"
    metadata_list = None       # E.g. ["actions1.json", "actions2.json"]
    output_dir = "samples_output_action_transformer_lr"
    device = "cuda"
    dtype = torch.bfloat16
    distributed = False
    num_frames = 49
    height = 352
    width = 640
    num_videos_to_generate = 100
    seed = 42


    vis = VisualizeResults(
        model_path=model_path,
        video_dir=video_dir,
        video_list_file=video_list_file,
        frames_path=frames_path,
        frames_list=frames_list,
        metadata_path=metadata_path,
        metadata_list=metadata_list,
        transformer_local_path=transformer_local_path,
        num_frames=num_frames,
        height=height,
        width=width,
        num_videos_to_generate=num_videos_to_generate,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
        distributed=distributed,
    )
    print(0)
    vis.run(seed=seed)

