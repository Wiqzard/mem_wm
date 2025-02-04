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
    num_frames = action_dict['wasd'].shape[1]
    action_names = list(action_dict.keys())
    
    formatted_lines = []
    for frame_idx in range(num_frames):
        active_actions = []
        
        # Process key actions
        if 'wasd' in action_dict:
            wasd_keys = ['w', 'a', 's', 'd']
            active_keys = [wasd_keys[i] for i in range(4) if action_dict['wasd'][0, frame_idx, i] > 0.5]
            active_actions.extend(active_keys)
        
        # Process other actions (space, shift, mouse buttons)
        for key in ['space', 'shift', 'mouse_1', 'mouse_2']:
            if key in action_dict and action_dict[key][0, frame_idx] > 0.5:
                active_actions.append(key.replace('_', ''))
        
        # Process dx and dy
        dx, dy = 0, 0
        if 'dx' in action_dict:
            dx = action_dict['dx'][0, frame_idx].item()
        if 'dy' in action_dict:
            dy = action_dict['dy'][0, frame_idx].item()
        
        if abs(dx) > 0 or abs(dy) > 0:
            active_actions.append(f'dx:{dx:.1f} dy:{dy:.1f}')
        
        # Format the frame output
        formatted_lines.append(f'Frame {frame_idx + 1}: {", ".join(active_actions)} |')
    
    return '\n'.join(formatted_lines)

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
            Path(str(video_path.parent / line.strip()).replace("videos/", "metadata/").replace(".mp4", ".json"))
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


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
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
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames


def load_actions_as_tensors(metadata_path: Path, num_actions: int) -> Dict[str, torch.Tensor]:
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
    #num_actions = len(actions)

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
