[![CI](https://github.com/huang3527/microseg-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/huang3527/microseg-vision/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# microseg-vision

A small, framework-like PyTorch project for 2D image segmentation.

The goal is to demonstrate:

- A clean project structure for segmentation work
- Reusable abstractions for data loading, transforms, models, training and inference
- Config-driven experiments using a simple YAML config

This repo is **completely generic** and uses only public datasets.
It is intended as a portfolio / demo project, not tied to any specific company or tool.

## Features

- Simple folder-based dataset loader (`images/` + `masks/`)
- Numpy/Pillow-based preprocessing and optional Torch transforms
- Minimal UNet implementation in PyTorch
- Config-driven training loop (`configs/unet_example.yaml`)
- Basic segmentation metrics (Dice, IoU, pixel accuracy)
- Inference utility for single images and folders

## Data layout

Expected directory structure:

```text
data/
  train/
    images/
      sample01.png
      sample02.png
      ...
    masks/
      sample01.png
      sample02.png
      ...
  val/
    images/
      ...
    masks/
      ...

---

## Installation

```bash
pip install -e .