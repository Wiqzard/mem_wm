import decord
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import json

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from finetune.datasets.utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_actions,
    load_videos,
    load_actions_as_tensors,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
    preprocess_video_with_resize_wm,
)


#if TYPE_CHECKING:
#    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)

class WMDataset(Dataset):
    """
    A unified dataset class that:

    1) Loads images/videos from paths, checks their existence.
    2) Optionally resizes videos/images to fixed dimensions if
       max_num_frames, height, and width are provided.
    3) Optionally loads a JSON metadata file containing actions if
       load_actions=True.
    4) Optionally encodes the video offline or on-the-fly using a
       trainer-provided encode function (encode_online switch).
    """

    def __init__(
        self,
        data_root: str,
        video_column: str,
        image_column: str | None,
        encode_online: bool = True,
        max_num_frames: int | None = 18,
        height: int | None = 64,
        width: int | None = 64,
        use_actions: bool = True,
        random_start: bool = True,
        name: str = "wm",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.use_actions = use_actions
        self.name = name
        self.random_start = random_start



        # Base dataset loading:
        self.videos = load_videos(self.data_root / video_column)

        if image_column is not None:
            self.images = load_images(self.data_root / image_column)
        else:
            self.images = load_images_from_videos(self.videos)

        self.actions = load_actions(self.data_root / video_column) 
        #self.actions = [load_actions(video_path) for video_path in self.videos]

        self.encode_online = encode_online

        # Quick checks:
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Missing video file: {next(path for path in self.videos if not path.is_file())}"
            )
        if any(not path.is_file() for path in self.images):
            raise ValueError(
                f"Missing image file: {next(path for path in self.images if not path.is_file())}"
            )
        if len(self.videos) != len(self.images):
            raise ValueError(
                f"Number of videos ({len(self.videos)}) and images ({len(self.images)}) do not match"
            )
        if len(self.videos) != len(self.actions):
            raise ValueError(
                f"Number of videos ({len(self.videos)}) and actions ({len(self.actions)}) do not match"
            )        


        # Define transforms if resizing is requested
        # (Simple per-pixel to [-1,1], can be more advanced if needed)
        self.frame_transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        ])
        # Same transform for images
        self.image_transform_fn = self.frame_transform

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int, num_frames: int | None = None) -> Dict[str, Any]:
        """
        Main data retrieval method. Loads images/videos, possibly transforms them,
        optionally loads actions, and returns a dictionary of Tensors.
        """
        # If you’re using a special bucket sampler that passes a list, handle that:
        if isinstance(index, list):
            return index

        video_path = self.videos[index]
        image_path = self.images[index]

        # Where to cache latents, etc.
        train_resolution_str = f"{self.max_num_frames}x{self.height}x{self.width}" 

        cache_dir = self.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / self.name / train_resolution_str
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        encoded_video_path = video_latent_dir / (video_path.stem + ".safetensors")

        # ----------------
        # 1) If we encode on the fly, just load and transform the raw frames
        # ----------------
        if self.encode_online:
            frames, image, start_index = self._preprocess(video_path, image_path, random_start=self.random_start, num_frames=num_frames)

            # save video to file "video.mp4"
            #import imageio
            #import numpy as np
            #fps = 16
            #writer = imageio.get_writer("video.mp4", fps=fps, codec="libx264")
            #for frame in frames:
            #    if not isinstance(frame, np.ndarray):
            #        frame = frame.numpy().transpose(1, 2, 0)  # Convert Decord frame to NumPy array if needed
            #    writer.append_data(frame)

            image = self.image_transform(image)
            frames = self.video_transform(frames)

            # Convert frames shape: [F, C, H, W] -> [B, C, F, H, W], B=1
            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

            data_dict = {
                "image": image,
                "video": frames,
                "encoded_video": None,
                "start_index": index,
                "video_metadata": {
                    "num_frames": frames.shape[2] // 4,
                    "height": frames.shape[3] // 8,
                    "width": frames.shape[4] // 8,
                },
            }

            # If we also want actions:
            if self.use_actions:
                action = self.actions[index]
                actions_tensor_dict = load_actions_as_tensors(
                    action, (self.max_num_frames or frames.shape[2]) - 1, start_index
                )
                data_dict["actions"] = actions_tensor_dict

            return data_dict

        # ----------------
        # 2) Offline-encoded path: check if we have latent file
        # ----------------
        if encoded_video_path.exists():
            # Load from existing latent
            loaded = load_file(encoded_video_path)
            encoded_video = loaded["encoded_video"]

            # Still need to load/transform the image
            _, image, _ = self._preprocess(None, image_path)
            image = self.image_transform(image)

            data_dict = {
                "image": image,
                "encoded_video": encoded_video,
                "video_metadata": {
                    "num_frames": encoded_video.shape[1],
                    "height": encoded_video.shape[2],
                    "width": encoded_video.shape[3],
                },
            }

            # If actions are requested
            if self.load_actions:
                # We do still need frames count to index the actions properly,
                # so let's set it to the number of frames in encoded_video
                start_index = 0
                metadata_path = self._get_metadata_path(video_path)
                actions_tensor_dict = self._load_actions_as_tensors(
                    metadata_path, encoded_video.shape[1], start_index
                )
                data_dict["actions"] = actions_tensor_dict

            return data_dict

    def _preprocess(
        self,
        video_path: Path | None,
        image_path: Path | None,
        random_start: bool = False,
        num_frames: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Consolidates the logic of loading and resizing:
          1) If max_num_frames is set, resize the video to (max_num_frames, height, width).
          2) If height,width is set, resize the image to (height, width).
          3) If `load_actions` indicates we’re using special logic (e.g. `_wm` version),
             then it calls that version. Otherwise, calls the normal version.
        Returns (frames, image, start_index).
        """
        frames = None
        image = None
        start_index = 0

        # --- Load Video ---
        if video_path is not None:
            # e.g. with watermark or random start:
            #   preprocess_video_with_resize_wm
            frames, first_frame, start_index = preprocess_video_with_resize_wm(
                video_path,
                max_num_frames=self.max_num_frames or num_frames,
                height=self.height,
                width=self.width,
                random_start=random_start,
            )
            image = first_frame
        return frames, image, start_index

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transforms to a 4D video tensor of shape [F, C, H, W].
        """
        # Example pixel-range transform to [-1,1].
        # If you need advanced transforms (e.g. cropping), add them.
        # We'll use the same transform for each frame.
        return torch.stack([self.frame_transform(f) for f in frames], dim=0)

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transforms to a 3D image tensor of shape [C, H, W].
        """
        return self.image_transform_fn(image)

    # -------------------------------------------------------------------------
    # Actions logic
    # -------------------------------------------------------------------------
    def _get_metadata_path(self, video_path: Path) -> Path:
        """
        Infers the metadata JSON file path from the video file path by
        replacing "videos" with "metadata" and ".mp4" with ".json".
        Adjust to match your dataset structure.
        """
        return Path(str(video_path).replace("videos", "metadata").replace(".mp4", ".json"))




if __name__ == "__main__":
    wm_datset = WMDataset(
        data_root="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc/processed/train_set/",
        video_column="videos_train_gf_processed.txt",
        image_column="images_train_gf_processed.txt",
        height=360, #256,
        width=480, #256,
        max_num_frames=49,
        encode_online=True
    )

    a = wm_datset[0]
    #print(wm_datset[1])
