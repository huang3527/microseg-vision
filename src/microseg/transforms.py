from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch


class Compose:
    """Compose multiple transforms for dict sample."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict):
        for t in self.transforms:
            sample = t(sample)
        return sample


@dataclass
class ToTensor:
    """Convert numpy arrays in sample to torch tensors."""

    def __call__(self, sample: Dict):
        image = sample["image"]  # (H, W, C)
        mask = sample.get("mask", None)

        image_t = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        sample["image"] = image_t

        if mask is not None:
            # mask: (H, W, 1) -> (1, H, W)
            mask_t = torch.from_numpy(mask).permute(2, 0, 1)
            sample["mask"] = mask_t

        return sample


@dataclass
class Normalize:
    """Normalize image to (image - mean) / std.

    mean, std in shape (C,) or scalar.
    """

    mean: float | Sequence[float] = 0.0
    std: float | Sequence[float] = 1.0

    def __call__(self, sample: Dict):
        img: torch.Tensor = sample["image"]
        mean_t = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device)
        std_t = torch.as_tensor(self.std, dtype=img.dtype, device=img.device)

        if mean_t.ndim == 0:
            mean_t = mean_t.view(1, 1, 1)
            std_t = std_t.view(1, 1, 1)
        elif mean_t.ndim == 1:
            mean_t = mean_t.view(-1, 1, 1)
            std_t = std_t.view(-1, 1, 1)

        sample["image"] = (img - mean_t) / std_t
        return sample


@dataclass
class RandomHorizontalFlip:
    p: float = 0.5

    def __call__(self, sample: Dict):
        if np.random.rand() < self.p:
            img: torch.Tensor = sample["image"]
            img = torch.flip(img, dims=[2])  # flip W
            sample["image"] = img

            mask = sample.get("mask", None)
            if mask is not None:
                mask = torch.flip(mask, dims=[2])
                sample["mask"] = mask
        return sample


@dataclass
class RandomVerticalFlip:
    p: float = 0.5

    def __call__(self, sample: Dict):
        if np.random.rand() < self.p:
            img: torch.Tensor = sample["image"]
            img = torch.flip(img, dims=[1])  # flip H
            sample["image"] = img

            mask = sample.get("mask", None)
            if mask is not None:
                mask = torch.flip(mask, dims=[1])
                sample["mask"] = mask
        return sample


@dataclass
class Resize:
    """Simple center-crop + resize using torch.interpolate."""

    size: tuple[int, int]  # (H, W)

    def __call__(self, sample: Dict):
        img: torch.Tensor = sample["image"].unsqueeze(0)  # (1, C, H, W)
        mask = sample.get("mask", None)
        if mask is not None:
            mask_t = mask.unsqueeze(0).float()
        else:
            mask_t = None

        img_resized = torch.nn.functional.interpolate(
            img, size=self.size, mode="bilinear", align_corners=False
        )
        sample["image"] = img_resized.squeeze(0)

        if mask_t is not None:
            mask_resized = torch.nn.functional.interpolate(
                mask_t, size=self.size, mode="nearest"
            )
            sample["mask"] = mask_resized.squeeze(0)

        return sample
