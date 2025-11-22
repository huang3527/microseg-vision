from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SegmentationDataset
from .metrics import compute_segmentation_metrics
from .models import UNetModel
from .transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)


class Trainer:
    """High-level training loop wrapper."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        model_cfg = config.get("model", {})
        self.model = UNetModel(model_cfg).to(self.device)

        train_cfg = config.get("train", {})
        self.epochs = int(train_cfg.get("epochs", 50))
        self.batch_size = int(train_cfg.get("batch_size", 4))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0))

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.output_dir = Path(config.get("output_dir", "runs/example"))
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _build_transforms(self, augment: bool) -> Compose:
        size = tuple(self.config.get("data", {}).get("image_size", [256, 256]))  # H, W

        tfs = [ToTensor(), Resize(size=size), Normalize(mean=0.5, std=0.5)]
        if augment:
            tfs.insert(1, RandomHorizontalFlip(p=0.5))
            tfs.insert(2, RandomVerticalFlip(p=0.5))
        return Compose(tfs)

    def _build_dataloaders(self):
        data_cfg = self.config.get("data", {})
        train_root = data_cfg.get("train_root", "data/train")
        val_root = data_cfg.get("val_root", "data/val")

        train_ds = SegmentationDataset(
            train_root, transform=self._build_transforms(augment=True)
        )
        val_ds = SegmentationDataset(
            val_root, transform=self._build_transforms(augment=False)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self._build_dataloaders()

        best_val_dice = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self._validate(val_loader)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"dice={val_metrics['dice']:.4f} "
                f"iou={val_metrics['iou']:.4f} "
                f"pixel_acc={val_metrics['pixel_acc']:.4f}"
            )

            # Save best checkpoint by Dice
            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                ckpt_path = self.output_dir / "checkpoints" / "best.pt"
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "config": self.config,
                        "epoch": epoch,
                        "val_dice": best_val_dice,
                    },
                    ckpt_path,
                )

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        return running_loss / len(loader.dataset)

    def _validate(self, loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        agg_metrics = {"dice": 0.0, "iou": 0.0, "pixel_acc": 0.0}
        n = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                metrics = compute_segmentation_metrics(logits, masks)
                bs = images.size(0)
                running_loss += loss.item() * bs
                for k in agg_metrics:
                    agg_metrics[k] += metrics[k] * bs
                n += bs

        for k in agg_metrics:
            agg_metrics[k] /= max(n, 1)
        return running_loss / max(n, 1), agg_metrics


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
