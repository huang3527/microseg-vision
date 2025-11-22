from __future__ import annotations

from typing import Dict

import torch


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Binary Dice on probabilities/logits already thresholded to {0,1}."""
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.bool()
    target = target.bool()
    correct = (pred == target).float().mean()
    return correct


def compute_segmentation_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Helper for binary segmentation metrics."""
    probs = logits.sigmoid()
    preds = (probs > threshold).float()

    dice = dice_coefficient(preds, target).item()
    iou = iou_score(preds, target).item()
    acc = pixel_accuracy(preds, target).item()
    return {"dice": dice, "iou": iou, "pixel_acc": acc}
