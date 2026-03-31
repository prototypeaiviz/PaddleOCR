"""
visualize_paddle_recaug.py

Visualize PaddleOCR built-in RecAug on ONE OCR image.

What this does:
- loads one image from the generated OCR dataset
- applies PaddleOCR's built-in RecAug multiple times
- saves a grid showing Original + several RecAug outputs

Important:
- augmentation logic comes from PaddleOCR, not from custom augment.py
- this is the cleanest way to demonstrate PaddleOCR augmentation visually
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def to_rgb(image_bgr):
    """Convert OpenCV BGR image to RGB for matplotlib."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main():
    project_root = Path(__file__).resolve().parents[1]
    paddleocr_root = Path.home() / "ocr_aug_work" / "PaddleOCR"

    # Make PaddleOCR repo importable
    sys.path.append(str(paddleocr_root))

    # Import PaddleOCR built-in RecAug
    from ppocr.data.imaug.rec_img_aug import RecAug

    # Pick one sample image from your dataset
    train_dir = project_root / "generated_dataset" / "train" / "images"
    image_paths = sorted(train_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {train_dir}")

    image_path = image_paths[0]
    print(f"Using image: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create PaddleOCR built-in augmenter
    rec_aug = RecAug(
        tia_prob=0.4,
        crop_prob=0.1,
        reverse_prob=0.0,
        noise_prob=0.2,
        jitter_prob=0.2,
        blur_prob=0.2,
        hsv_aug_prob=0.2,
    )

    # Build outputs:
    # Original + several random RecAug results on the SAME image
    visuals = [("Original", image_bgr.copy())]

    # Each call to RecAug is random, so you get different versions
    for i in range(1, 9):
        sample = {"image": image_bgr.copy()}
        out = rec_aug(sample)

        if out is None or "image" not in out:
            raise RuntimeError("RecAug did not return a valid image.")

        visuals.append((f"RecAug #{i}", out["image"]))

    # Plot grid
    cols = 3
    rows = (len(visuals) + cols - 1) // cols

    plt.figure(figsize=(14, 4 * rows))

    for idx, (title, img_bgr) in enumerate(visuals, start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(to_rgb(img_bgr))
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.suptitle("PaddleOCR Built-in RecAug Visualization", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = project_root / "generated_dataset" / "paddle_recaug_demo.png"
    plt.savefig(output_path, dpi=180)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()