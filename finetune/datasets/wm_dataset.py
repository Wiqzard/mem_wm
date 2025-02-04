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

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseI2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        encode_online: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        print("dataset initializing")
        data_root = Path(data_root)
        # self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        if image_column is not None:
            self.images = load_images(data_root / image_column)
        else:
            self.images = load_images_from_videos(self.videos)
        self.trainer = trainer

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.encode_online = encode_online 

        # Check if number of prompts matches number of videos and images
        # if not (len(self.videos) == len(self.prompts) == len(self.images)):
        #    raise ValueError(
        #        f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=} and {len(self.images)=}. Please ensure that the number of caption prompts, videos and images match in your dataset."
        #    )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if all image files exist
        if any(not path.is_file() for path in self.images):
            raise ValueError(
                f"Some image files were not found. Please ensure that all image files exist in the dataset directory. Missing file: {next(path for path in self.images if not path.is_file())}"
            )
        print("dataset initialized")

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        # prompt = self.prompts[index]
        video = self.videos[index]
        image = self.images[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        #video_latent_dir = (
        #    cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        #)
        video_latent_dir = (
            cache_dir / "video_latent" / "cogvideox1.5-i2v-wm" / train_resolution_str
        )
        # prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        # prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        # prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        # prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")

        # if prompt_embedding_path.exists():
        #    prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        #    logger.debug(
        #        f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
        #        main_process_only=False,
        #    )
        # else:
        #    prompt_embedding = self.encode_text(prompt)
        #    prompt_embedding = prompt_embedding.to("cpu")
        #    # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
        #    prompt_embedding = prompt_embedding[0]
        #    save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
        #    logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)
        if self.encode_online:
            frames, image = self.preprocess(video, image)
            image = self.image_transform(image)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)
            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            return {
                "image": image,
                "video": frames,
                "encoded_video": None,
                "video_metadata": {
                    "num_frames": frames.shape[2] // 4, #// self.encoder vae ...
                    "height": frames.shape[3] // 8 ,#// self.encoder vae ...
                    "width": frames.shape[4] // 8 , #// self.encoder vae ...
                },
                
            }


        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(
                f"Loaded encoded video from {encoded_video_path}", main_process_only=False
            )
            # shape of image: [C, H, W]
            _, image = self.preprocess(None, self.images[index])
            image = self.image_transform(image)
        else:
            logger.debug(f"video path: {encoded_video_path}", main_process_only=False)
            frames, image = self.preprocess(video, image)
            frames = frames.to(self.device)
            image = image.to(self.device)
            image = self.image_transform(image)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)

            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = image.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image,
            # "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(
                video_path, self.max_num_frames, self.height, self.width
            )
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithActions(I2VDatasetWithResize):
    """
    A dataset class for image-to-video generation that:
      1) Resizes inputs to a fixed dimension (inherited from I2VDatasetWithResize).
      2) Loads a JSON metadata file containing "actions" for each video.

    The metadata file is assumed to live under a 'metadata/' directory,
    with the same filename as the corresponding video, but a '.json' extension.

    E.g., if a video is:
        /path/to/videos/myvideo.mp4
    The metadata file is expected at:
        /path/to/metadata/myvideo.json

    The JSON structure is expected to have a top-level key "actions"
    that is a list of objects, each with:
        {
          "dx": <float>,
          "dy": <float>,
          "buttons": <list of int>,
          "keys": <list of str>
        }

    As output, __getitem__ returns a dictionary that includes:
      "actions": A dictionary of Tensors with shapes:
            "wasd":    (B, T, 4)
            "space":   (B, T)
            "shift":   (B, T)
            "mouse_1": (B, T)
            "mouse_2": (B, T)
            "dx":      (B, T)
            "dy":      (B, T)
        where B=1 for a single sample, and T is the number of actions loaded.
    """

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # First, get the base dictionary from I2VDatasetWithResize
        data = super().__getitem__(index)

        # The above 'data' dictionary contains:
        #   {
        #       "image": image_tensor,
        #       "prompt_embedding": prompt_embedding_tensor,
        #       "encoded_video": encoded_video_tensor,
        #       "video_metadata": {...}
        #   }

        # Now, load the action metadata:
        video_path = self.videos[index]  # e.g., /path/to/videos/<name>.mp4
        metadata_path = self._get_metadata_path(video_path)
        actions_tensor_dict = self._load_actions_as_tensors(
            metadata_path, self.max_num_frames - 1 #data["video_metadata"]["num_frames"] * 8
        )

        data["actions"] = actions_tensor_dict
        return data

    def _get_metadata_path(self, video_path: Path) -> Path:
        """
        Infers the metadata JSON file path from the video file path.
        For example, changing:
            /.../videos/<name>.mp4
        to:
            /.../metadata/<name>.json
        """
        # Example approach:
        # 1. Start from the parent directory of the videos folder.
        # 2. Replace "videos" with "metadata" in the path.
        # 3. Replace the .mp4 extension with .json.
        #    (you can adjust if your naming pattern is different)

        # parent_of_videos_dir = video_path.parent.parent
        # metadata_folder = parent_of_videos_dir / "metadata"
        # same_filename_stem = video_path.stem
        # return metadata_folder / f"{same_filename_stem}.json"

        # Or a simpler string-based approach if you rely on a fixed structure:
        # (Be sure to handle the case if the video_path does not follow the pattern.)
        # This is just an example—adapt it to your dataset layout:
        return Path(str(video_path).replace("/videos/", "/metadata/").replace(".mp4", ".json"))

    def _load_actions_as_tensors(
        self, metadata_path: Path, num_actions: int = 10000
    ) -> Dict[str, torch.Tensor]:
        """
        Loads the JSON metadata from `metadata_path` and converts it into Tensors
        with shapes:

          wasd:    (1, T, 4)
          space:   (1, T)
          shift:   (1, T)
          mouse_1: (1, T)
          mouse_2: (1, T)
          dx:      (1, T)
          dy:      (1, T)

        Returns them as a dict under the key "actions".
        """
        if not metadata_path.exists():
            # If metadata is missing, handle gracefully or raise error
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        actions = metadata["actions"]  # list of dicts with "dx", "dy", "buttons", "keys"
        num_actions_in_sample = len(actions)
        num_actions = min(
            num_actions_in_sample, num_actions 
        )  # Limit to num_actions, or pad if fewer

        # Prepare empty torch arrays:
        # (T,) or (T,4), but eventually we add a batch dim => (1,T,...) for the final return
        wasd = torch.zeros(num_actions, 4, dtype=torch.float)
        space = torch.zeros(num_actions, dtype=torch.float)
        shift = torch.zeros(num_actions, dtype=torch.float)
        mouse_1 = torch.zeros(num_actions, dtype=torch.float)
        mouse_2 = torch.zeros(num_actions, dtype=torch.float)
        dx = torch.zeros(num_actions, dtype=torch.float)
        dy = torch.zeros(num_actions, dtype=torch.float)

        # We'll map 'w','a','s','d' to indices 0..3
        wasd_map = {"w": 0, "a": 1, "s": 2, "d": 3}

        for t, action in enumerate(actions):
            if t >= num_actions:
                break
            # dx, dy
            dx[t] = float(action.get("dx", 0.0))
            dy[t] = float(action.get("dy", 0.0))

            # keys: e.g. ["w", "shift", "space"]
            keys = action.get("keys", [])
            for k in keys:
                if k in wasd_map:
                    wasd[t, wasd_map[k]] = 1.0
                elif k == "space":
                    space[t] = 1.0
                elif k == "shift":
                    shift[t] = 1.0
                # If you have more keys you care about, add logic here

            # buttons: e.g. [0] or [0,1] or [1] or []
            buttons = action.get("buttons", [])
            if 0 in buttons:  # Typically mouse left
                mouse_1[t] = 1.0
            if 1 in buttons:  # Typically mouse right
                mouse_2[t] = 1.0

        # Insert batch dimension of size 1 => (1, T, ...)
        actions_tensor_dict = {
            "wasd": wasd.unsqueeze(0),  # (1, T, 4)
            "space": space.unsqueeze(0),  # (1, T)
            "shift": shift.unsqueeze(0),  # (1, T)
            "mouse_1": mouse_1.unsqueeze(0),  # (1, T)
            "mouse_2": mouse_2.unsqueeze(0),  # (1, T)
            "dx": dx.unsqueeze(0),  # (1, T)
            "dy": dy.unsqueeze(0),  # (1, T)
        }
        #print(wasd.shape)
        #print(space.shape)
        #print(shift.shape)
        #print(mouse_1.shape)
        #print(dx.shape)
        return actions_tensor_dict


class I2VDatasetWithBuckets(BaseI2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
