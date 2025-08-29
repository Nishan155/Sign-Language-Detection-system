import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.models.video import r2plus1d_18
from concurrent.futures import ThreadPoolExecutor

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained R(2+1)D model and remove final classifier
model = r2plus1d_18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Output: (B, 512, 1, 1, 1)
model.to(device).eval()

# ImageNet normalization
normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])

# Augment a single frame
def augment_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = F.to_pil_image(frame)

    if random.random() > 0.5:
        frame = F.hflip(frame)

    frame = F.resize(frame, (256, 256))
    i, j, h, w = transforms.RandomCrop.get_params(frame, output_size=(112, 112))
    frame = F.crop(frame, i, j, h, w)

    frame = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)(frame)
    return normalize(F.to_tensor(frame))

# Process a single frame
def process_frame(frame: np.ndarray, spatial_aug=False) -> torch.Tensor:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    if spatial_aug:
        return augment_frame(tensor_frame)
    else:
        resized = F.resize(F.to_pil_image(tensor_frame), (112, 112))
        return normalize(F.to_tensor(resized))

# Extract N uniformly spaced frames from the video
def extract_frames(video_path: str, target_frames: int = 16, every_n: int = 4,
                   temporal_jitter: bool = False, spatial_aug: bool = False) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    # Temporal jitter
    step = every_n + random.choice([-1, 0, 1]) if temporal_jitter else every_n
    sampled = [frames[i] for i in range(0, len(frames), step)]

    # Pad or truncate to fixed length
    if len(sampled) < target_frames:
        sampled += [sampled[-1]] * (target_frames - len(sampled))
    else:
        sampled = sampled[:target_frames]

    # Apply transforms in parallel
    with ThreadPoolExecutor() as executor:
        processed = list(executor.map(lambda f: process_frame(f, spatial_aug), sampled))

    return torch.stack(processed).permute(1, 0, 2, 3).unsqueeze(0)  # Shape: (1, 3, T, H, W)

# Extract R(2+1)D features from a video
def extract_video_feature(video_path: str, temporal_jitter=False, spatial_aug=False) -> np.ndarray:
    frames = extract_frames(video_path, target_frames=16, temporal_jitter=temporal_jitter, spatial_aug=spatial_aug).to(device)
    with torch.no_grad():
        features = model(frames)  # (1, 512, 1, 1, 1)
    return features.view(-1).cpu().numpy()  # (512,)