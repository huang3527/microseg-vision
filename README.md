# microseg-vision

[![CI](https://github.com/huang3527/microseg-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/huang3527/microseg-vision/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/huang3527/microseg-vision/branch/main/graph/badge.svg)](https://codecov.io/gh/huang3527/microseg-vision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

<p align="center">
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png" width="160">
</p>

<p align="center">
<b>A clean, minimal, and production-ready PyTorch pipeline for 2D image segmentation.</b><br>
Designed for reproducible research, professional portfolio, and educational tutorials.
</p>

---

## 🚀 Features

- 📁 Simple **folder-based segmentation dataset** (`images/` + `masks/`)
- 🔄 Modular **Transforms** (Resize, H/V Flip, Normalize, ToTensor)
- 🧠 Clean **UNet implementation** in pure PyTorch
- ⚙️ **YAML config system** for experiment reproducibility
- 🏋️ **Trainer** with validation metrics (Dice/IoU/Pixel Accuracy)
- 🖼 **InferenceEngine** for single images or whole folders
- 📦 Publish-ready project structure (`pyproject.toml`, MIT License)
- 🔧 GitHub Actions CI (lint, test, style)
- 📊 Optional Codecov coverage support

---

## 📘 Example: Training on Kvasir-SEG (public dataset)

The **Kvasir-SEG** dataset contains 1,000+ colonoscopy images with masks.  
It is fully open-source and ideal for demonstrating segmentation pipelines.

Dataset link:  
https://datasets.simula.no/kvasir-seg/

---

### 1. Prepare Data

mkdir -p data/train data/val

Place data into:
data/
  train/
    images/
    masks/
  val/
    images/
    masks/

### 2. Train
python -m microseg.train --config configs/unet_example.yaml

Results + checkpoints will be saved to:
runs/example/checkpoints/best.pt

### 3. Inference

python -m microseg.infer \
  --config configs/unet_example.yaml \
  --checkpoint runs/example/checkpoints/best.pt \
  --input_folder path/to/images \
  --output_folder path/to/output_masks

The output masks will be saved as binary .png.

### 📂 Project Structure

microseg-vision/
├── configs/
│   └── unet_example.yaml
├── experiments/
│   └── notebooks/
│       └── demo.ipynb
├── src/
│   └── microseg/
│       ├── data.py
│       ├── transforms.py
│       ├── models.py
│       ├── metrics.py
│       ├── train.py
│       └── infer.py
├── tests/
│   └── test_import.py
├── .github/workflows/ci.yml
├── LICENSE
├── pyproject.toml
├── README.md
└── Makefile

### 🛠 Development

make install
make lint
make test
make train
make infer

### 🧪 Coverage

Install extra tools:
pip install pytest-cov codecov

Run:
pytest --cov=src/microseg --cov-report=xml

Upload coverage (GitHub Actions does this automatically):
codecov

###  📜 License

MIT License — You are free to use, modify, and distribute this project.

### ⭐ Acknowledgments

	•	Kvasir-SEG dataset (Simula)
	•	PyTorch team
	•	Open-source contributors