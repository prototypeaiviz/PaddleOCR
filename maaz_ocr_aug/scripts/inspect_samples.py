"""Visual inspection tool for generated OCR samples.

This script builds a simple contact sheet so you can quickly check if the
augmentation is still producing readable text.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_dir = args.dataset_dir / "train" / "images"
    images = sorted(image_dir.glob("*.png"))[: args.count]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    cols = 3
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=(12, 3 * rows))

    for i, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i)
        plt.imshow(image)
        plt.title(image_path.name)
        plt.axis("off")

    out_path = args.dataset_dir / "sample_grid.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved inspection sheet to: {out_path}")
