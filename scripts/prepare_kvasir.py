"""
Download Kvasir-SEG dataset and prepare it into:

data/
  train/
    images/
    masks/
  val/
    images/
    masks/

Default split: 80% train / 20% val.
"""

from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path
from typing import Tuple

import requests
from tqdm import tqdm

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[info] File already exists, skip download: {dst}")
        return

    print(f"[info] Downloading {url} -> {dst}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dst, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def extract_zip(zip_path: Path, dst_dir: Path) -> Path:
    print(f"[info] Extracting {zip_path} -> {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)
    # Kvasir-SEG usually extracts to something like dst_dir / "kvasir-seg"
    # Try to detect it:
    for child in dst_dir.iterdir():
        if child.is_dir() and "kvasir" in child.name.lower():
            print(f"[info] Found dataset root: {child}")
            return child
    raise RuntimeError(f"Could not find dataset root under {dst_dir}")


def prepare_split(
    raw_root: Path,
    out_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """
    Kvasir-SEG structure (after extract) is something like:

    raw_root/
      images/
        xxx.png
      masks/
        xxx.png
    """
    images_dir = raw_root / "images"
    masks_dir = raw_root / "masks"

    if not images_dir.is_dir():
        # Some versions put images/masks one folder deeper.
        # Try to search for them.
        candidates = list(raw_root.rglob("images"))
        if not candidates:
            raise FileNotFoundError(f"Could not find 'images' folder under {raw_root}")
        images_dir = candidates[0]
        masks_dir = images_dir.parent / "masks"

    assert masks_dir.is_dir(), f"masks dir not found: {masks_dir}"

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    print(f"[info] Found {len(image_files)} images")

    random.seed(seed)
    random.shuffle(image_files)

    n_train = int(len(image_files) * train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]

    def copy_split(files, split: str):
        img_out = out_root / split / "images"
        mask_out = out_root / split / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(files, desc=f"Copying {split}"):
            stem = img_path.stem
            mask_path = masks_dir / f"{stem}{img_path.suffix}"
            if not mask_path.exists():
                # Some datasets might use different extensions for masks (e.g. .jpg vs .png)
                # Try common alternatives:
                alt_candidates = [
                    masks_dir / f"{stem}.png",
                    masks_dir / f"{stem}.jpg",
                    masks_dir / f"{stem}.jpeg",
                ]
                mask_path = None
                for c in alt_candidates:
                    if c.exists():
                        mask_path = c
                        break

            if mask_path is None or not mask_path.exists():
                print(f"[warn] Mask not found for {img_path.name}, skip")
                continue

            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(mask_path, mask_out / mask_path.name)

    copy_split(train_files, "train")
    copy_split(val_files, "val")

    print(f"[done] Prepared dataset at {out_root}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    raw_dir = data_root / "raw_kvasir"
    zip_path = data_root / "kvasir-seg.zip"

    # 1) download
    download_file(KVASIR_URL, zip_path)

    # 2) extract
    dataset_root = extract_zip(zip_path, raw_dir)

    # 3) prepare train/val split
    out_root = data_root
    prepare_split(dataset_root, out_root, train_ratio=0.8, seed=42)


if __name__ == "__main__":
    main()
