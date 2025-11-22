from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    image: np.ndarray  # (H, W, C) float32 [0, 1]
    mask: Optional[np.ndarray]  # (H, W) or (H, W, 1), 0/1
    id: str


class SegmentationDataset(Dataset):
    """Simple folder-based segmentation dataset.

    Expected layout:

        root/
          images/
            xxx.png
          masks/
            xxx.png

    The `transform` is applied to a dict {"image": image, "mask": mask, "id": id}.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.transform = transform
        self.image_suffixes = image_suffixes

        self._items: List[tuple[Path, Optional[Path], str]] = []
        self._scan()

    def _scan(self) -> None:
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        mask_exists = self.masks_dir.is_dir()

        for p in sorted(self.images_dir.iterdir()):
            if p.suffix.lower() not in self.image_suffixes:
                continue
            sample_id = p.stem
            mask_path: Optional[Path] = None
            if mask_exists:
                candidate = self.masks_dir / (sample_id + p.suffix)
                if candidate.is_file():
                    mask_path = candidate
            self._items.append((p, mask_path, sample_id))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path, sample_id = self._items[idx]

        image = Image.open(img_path).convert("L")  # default single-channel
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=-1)  # (H, W, 1)

        mask_np: Optional[np.ndarray] = None
        if mask_path is not None and mask_path.is_file():
            mask = Image.open(mask_path).convert("L")
            mask_np = np.asarray(mask, dtype=np.float32) / 255.0
            mask_np = (mask_np > 0.5).astype(np.float32)  # binarize
            mask_np = np.expand_dims(mask_np, axis=-1)  # (H, W, 1)

        sample = {"image": image_np, "mask": mask_np, "id": sample_id}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # Optional helper if you ever want to iterate over Sample dataclasses
    def iter_samples(self) -> Iterator[Sample]:
        for i in range(len(self)):
            d = self[i]
            yield Sample(
                image=d["image"],
                mask=d["mask"],
                id=d["id"],
            )

    def __iter__(self) -> Iterable[Sample]:
        return self.iter_samples()
