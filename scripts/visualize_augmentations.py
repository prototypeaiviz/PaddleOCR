"""
visualize_augmentations.py

Create a stronger OCR-focused augmentation visualization.

What this script shows
----------------------
1. The original OCR crop
2. Each augmentation individually
3. A few realistic sequential OCR-style augmentation pipelines

Why this is useful
------------------
Your supervisor wanted to clearly distinguish:
- the original image / ground truth
- the different augmented versions

This script makes that comparison much clearer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# Make project root importable so `src/augment.py` can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.augment import OCRAugmentor  # noqa: E402


def to_rgb(image_bgr):
    """Convert OpenCV BGR image to RGB for matplotlib display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_visualizations(image_bgr, augmenter: OCRAugmentor):
    """
    Build a list of named visualization outputs.

    We include:
    - single augmentations
    - sequential OCR-style augmentations
    """
    visuals = []

    # Original
    visuals.append(("Original", image_bgr.copy()))

    # Individual augmentations
    visuals.append(("Rotation", augmenter.random_rotation(image_bgr.copy())))
    visuals.append(("Perspective", augmenter.random_perspective(image_bgr.copy())))
    visuals.append(("Stretch", augmenter.horizontal_stretch(image_bgr.copy())))
    visuals.append(("Blur", augmenter.gaussian_blur(image_bgr.copy())))
    visuals.append(("Noise", augmenter.add_gaussian_noise(image_bgr.copy())))
    visuals.append(("Brightness/Contrast", augmenter.adjust_brightness_contrast(image_bgr.copy())))

    # Sequential OCR-style pipelines
    seq1 = image_bgr.copy()
    seq1 = augmenter.random_perspective(seq1)
    seq1 = augmenter.random_rotation(seq1)
    seq1 = augmenter.gaussian_blur(seq1)
    visuals.append(("Seq: Perspective → Rotation → Blur", seq1))

    seq2 = image_bgr.copy()
    seq2 = augmenter.horizontal_stretch(seq2)
    seq2 = augmenter.add_gaussian_noise(seq2)
    seq2 = augmenter.adjust_brightness_contrast(seq2)
    visuals.append(("Seq: Stretch → Noise → Brightness", seq2))

    seq3 = image_bgr.copy()
    seq3 = augmenter.random_rotation(seq3)
    seq3 = augmenter.gaussian_blur(seq3)
    seq3 = augmenter.add_gaussian_noise(seq3)
    visuals.append(("Seq: Rotation → Blur → Noise", seq3))

    return visuals


def main():
    # Adjust this path if needed to match your dataset structure
    train_dir = Path("generated_dataset/train/images")
    image_paths = sorted(train_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {train_dir}")

    # Use the first sample for reproducibility
    image_path = image_paths[0]
    print(f"Using image: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    augmenter = OCRAugmentor()

    visuals = build_visualizations(image_bgr, augmenter)

    # Grid layout
    cols = 3
    rows = (len(visuals) + cols - 1) // cols

    plt.figure(figsize=(15, 4 * rows))

    for idx, (title, img_bgr) in enumerate(visuals, start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(to_rgb(img_bgr))
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.suptitle("OCR Augmentation Visualization (Original vs Individual vs Sequential)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path("generated_dataset/augmentation_demo_better.png")
    plt.savefig(output_path, dpi=180)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()