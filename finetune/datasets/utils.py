import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import torch
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########


def format_action_string(action_dict):
    num_frames = action_dict["w"].shape[1]
    action_names = list(action_dict.keys())

    formatted_lines = []
    for frame_idx in range(num_frames):
        active_actions = []

        # Process key actions

        wasd_keys = ["w", "a", "s", "d"]
        for key in wasd_keys:
            if key in action_dict:
                if action_dict[key][0, frame_idx] > 0.5:
                    active_actions.append(key)

        # Process other actions (space, shift, mouse buttons)
        for key in ["space", "shift", "mouse_1", "mouse_2"]:
            if key in action_dict and action_dict[key][0, frame_idx] > 0.5:
                active_actions.append(key.replace("_", ""))

        # Process dx and dy
        dx, dy = 0, 0
        if "dx" in action_dict:
            dx = action_dict["dx"][0, frame_idx].item()
        if "dy" in action_dict:
            dy = action_dict["dy"][0, frame_idx].item()

        if abs(dx) > 0 or abs(dy) > 0:
            active_actions.append(f"dx:{dx:.1f} dy:{dy:.1f}")

        # Format the frame output
        formatted_lines.append(f'T {frame_idx + 1}: {", ".join(active_actions)} |')

    return " | ".join(formatted_lines)


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [
            image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_actions(video_path: Path) -> List[str]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            Path(
                str(video_path.parent / line.strip())
                .replace("videos", "metadata")
                .replace(".mp4", ".json")
            )
            for line in file.readlines()
            if len(line.strip()) > 0
        ]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########


def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image


def preprocess_video_with_resize_wm(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
    random_start: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads, resizes, and returns all frames of a video (up to max_num_frames)
    plus the first frame used (which corresponds to the actual start index).

    Returns:
        frames: torch.Tensor of shape [F, C, H, W].
        first_frame: torch.Tensor of shape [C, H, W] corresponding
                     to frames[0].
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)

    # Create a decord VideoReader with on-the-fly resizing
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)

    # -- Fewer frames than max_num_frames: replicate last frame --
    if video_num_frames < max_num_frames:
        frames = video_reader.get_batch(list(range(video_num_frames)))  # shape: [num, H, W, C]
        last_frame = frames[-1:]
        repeats_needed = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(repeats_needed, 1, 1, 1)  # replicate last frame
        frames = torch.cat([frames, repeated_frames], dim=0)  # [max_num_frames, H, W, C]

        frames = frames.float().permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
        first_frame = frames[0]  # [C, H, W]
        return frames, first_frame, 0

    # -- Otherwise, video has enough frames --
    if random_start:
        # Pick a random start so that we can still read max_num_frames
        start_frame_idx = torch.randint(0, video_num_frames - max_num_frames + 1, (1,)).item()
    else:
        start_frame_idx = 0

    indices = list(range(start_frame_idx, start_frame_idx + max_num_frames))
    frames = video_reader.get_batch(indices)  # [max_num_frames, H, W, C]
    frames = frames.float().permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
    first_frame = frames[0]  # [C, H, W]
    return frames, first_frame, start_frame_idx


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        if random_start:
            start_frame = torch.randint(0, video_num_frames - max_num_frames + 1, (1,)).item()
            indices = list(range(start_frame, start_frame + max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[:max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            return frames, indices
        else:
            indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))

            frames = video_reader.get_batch(indices)
            frames = frames[:max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            return frames


def load_actions_as_tensors(
    metadata_path: Path, num_actions: int, start_index: int = 0, action_list: List = None
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

    Returns them in a dict under the key "actions".
    """
    if metadata_path is not None:
        if not metadata_path.exists():
            # If metadata is missing, handle gracefully or raise error
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        actions = metadata["actions"]  # list of dicts with "dx", "dy", "buttons", "keys"

    if action_list is not None:
        actions = action_list

    actions = metadata["actions"]  # list of dicts
    num_actions_in_sample = len(actions)
    total_actions = min(num_actions_in_sample, num_actions)
    if total_actions < num_actions:
        logging.warning(
            f"Total actions in sample ({num_actions_in_sample}) is less than requested ({num_actions})"
        )

    w = torch.zeros(total_actions, dtype=torch.float)
    a = torch.zeros(total_actions, dtype=torch.float)
    s = torch.zeros(total_actions, dtype=torch.float)
    d = torch.zeros(total_actions, dtype=torch.float)
    e = torch.zeros(total_actions, dtype=torch.float)
    esc = torch.zeros(total_actions, dtype=torch.float)
    dwheel = torch.zeros(total_actions, dtype=torch.float)
    space = torch.zeros(total_actions, dtype=torch.float)
    shift = torch.zeros(total_actions, dtype=torch.float)
    mouse_1 = torch.zeros(total_actions, dtype=torch.float)
    mouse_2 = torch.zeros(total_actions, dtype=torch.float)
    dx = torch.zeros(total_actions, dtype=torch.float)
    dy = torch.zeros(total_actions, dtype=torch.float)

    count = 0
    # for t, action in enumerate(actions):
    for t, action in zip(range(start_index, start_index + total_actions), actions[start_index:]):
        # Skip until we reach start_index
        if t < start_index:
            continue
        if count >= total_actions:
            break

        dx[count] = float(action.get("dx", 0.0))
        dy[count] = float(action.get("dy", 0.0))

        # Process keys
        for k in action.get("keys", []):
            if k == "w":
                w[count] = 1.0
            elif k == "a":
                a[count] = 1.0
            elif k == "s":
                s[count] = 1.0
            elif k == "d":
                d[count] = 1.0
            elif k == "e":
                e[count] = 1.0
            elif k == "esc":
                esc[count] = 1.0
            elif k == "space":
                space[count] = 1.0
            elif k == "shift":
                shift[count] = 1.0

        # Process mouse buttons
        for b in action.get("buttons", []):
            if b == 0:
                mouse_1[count] = 1.0
            elif b == 1:
                mouse_2[count] = 1.0

        count += 1

    actions_tensor_dict = {
        "w": w.unsqueeze(0),
        "a": a.unsqueeze(0),
        "s": s.unsqueeze(0),
        "d": d.unsqueeze(0),
        "e": e.unsqueeze(0),
        "esc": esc.unsqueeze(0),
        "space": space.unsqueeze(0),
        "dwheel": dwheel.unsqueeze(0),
        "shift": shift.unsqueeze(0),
        "mouse_1": mouse_1.unsqueeze(0),
        "mouse_2": mouse_2.unsqueeze(0),
        "dx": dx.unsqueeze(0),
        "dy": dy.unsqueeze(0),
    }
    return actions_tensor_dict


def preprocess_video_with_buckets(
    video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file.
        resolution_buckets: List of tuples (num_frames, height, width) representing
            available resolution buckets.

    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    resolution_buckets = [bucket for bucket in resolution_buckets if bucket[0] <= video_num_frames]
    if len(resolution_buckets) == 0:
        raise ValueError(
            f"video frame count in {video_path} is less than all frame buckets {resolution_buckets}"
        )

    nearest_frame_bucket = min(
        resolution_buckets,
        key=lambda bucket: video_num_frames - bucket[0],
        default=1,
    )[0]
    frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))
    frames = video_reader.get_batch(frame_indices)
    frames = frames[:nearest_frame_bucket].float()
    frames = frames.permute(0, 3, 1, 2).contiguous()

    nearest_res = min(
        resolution_buckets, key=lambda x: abs(x[1] - frames.shape[2]) + abs(x[2] - frames.shape[3])
    )
    nearest_res = (nearest_res[1], nearest_res[2])
    frames = torch.stack([resize(f, nearest_res) for f in frames], dim=0)

    return frames
