from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image

from .models import UNetModel
from .transforms import Compose, Normalize, Resize, ToTensor


class InferenceEngine:
    """Run segmentation on images or folders."""

    def __init__(self, model_path: str | Path, config: Dict[str, Any]):
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config

        model_cfg = config.get("model", {})
        self.model = UNetModel(model_cfg).to(self.device)
        self._load_weights(model_path)

        size = tuple(config.get("data", {}).get("image_size", [256, 256]))
        self.transform = Compose(
            [ToTensor(), Resize(size=size), Normalize(mean=0.5, std=0.5)]
        )

    def _load_weights(self, model_path: str | Path):
        ckpt = torch.load(model_path, map_location=self.device)
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def segment_image(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary mask for one image, as numpy array (H, W)."""
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # (H, W, 1)

        sample = {"image": image, "mask": None, "id": "inference"}
        sample = self.transform(sample)
        img_t = sample["image"].unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            logits = self.model(img_t)
            probs = logits.sigmoid()
            preds = (probs > threshold).float()

        mask = preds.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        return mask

    def segment_folder(
        self,
        input_root: str | Path,
        output_root: str | Path,
        image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ):
        input_root = Path(input_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(input_root.iterdir()):
            if img_path.suffix.lower() not in image_suffixes:
                continue
            image = Image.open(img_path).convert("L")
            image_np = np.asarray(image, dtype=np.float32) / 255.0

            mask = self.segment_image(image_np)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(output_root / img_path.name)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = InferenceEngine(args.checkpoint, cfg)
    engine.segment_folder(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
