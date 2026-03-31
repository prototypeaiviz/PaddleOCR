"""Create a synthetic OCR word dataset with optional custom augmentation.

This script is useful for learning the pipeline end-to-end in VS Code.
It generates simple word images, optionally augments them, and writes
PaddleOCR-style train/val label files.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# Add local src/ to Python path when running from the project root.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from augment import AugmentConfig, OCRAugmentor
from dataset_utils import load_words, render_word_image, write_label_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--word_file", type=Path, default=PROJECT_ROOT / "data/words/words.txt")
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=300)
    parser.add_argument("--image_height", type=int, default=32)
    parser.add_argument("--min_width", type=int, default=96)
    parser.add_argument("--max_width", type=int, default=256)
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable custom augmentation during dataset generation",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pil_to_bgr_np(pil_image):
    """Convert a PIL image to OpenCV BGR format."""
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_np_to_rgb_np(image_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR back to RGB for saving with cv2 or PIL if needed."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def generate_split(
    split_name: str,
    num_samples: int,
    output_dir: Path,
    words: list[str],
    augmentor: OCRAugmentor | None,
    image_height: int,
    min_width: int,
    max_width: int,
) -> None:
    """Generate one dataset split (train or val)."""
    split_dir = output_dir / split_name
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []

    for idx in range(num_samples):
        label = random.choice(words)

        # Render a clean text image first.
        pil_img = render_word_image(
            text=label,
            image_height=image_height,
            min_width=min_width,
            max_width=max_width,
        )
        image = pil_to_bgr_np(pil_img)

        # Important design choice:
        # - We usually augment only the TRAIN split.
        # - Validation should stay clean so we can measure real performance.
        if split_name == "train" and augmentor is not None:
            image = augmentor(image)

        file_name = f"{split_name}_{idx:06d}.png"
        abs_path = image_dir / file_name
        rel_path = f"images/{file_name}"

        cv2.imwrite(str(abs_path), image)
        rows.append((rel_path, label))

    label_file = split_dir / f"{split_name}.txt"
    write_label_file(label_file, rows)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    words = load_words(args.word_file)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentor = None
    if not args.no_augment:
        augmentor = OCRAugmentor(AugmentConfig())

    generate_split(
        split_name="train",
        num_samples=args.num_train,
        output_dir=output_dir,
        words=words,
        augmentor=augmentor,
        image_height=args.image_height,
        min_width=args.min_width,
        max_width=args.max_width,
    )

    generate_split(
        split_name="val",
        num_samples=args.num_val,
        output_dir=output_dir,
        words=words,
        augmentor=None,
        image_height=args.image_height,
        min_width=args.min_width,
        max_width=args.max_width,
    )

    print(f"Dataset created at: {output_dir}")
    print(f"Train labels: {output_dir / 'train' / 'train.txt'}")
    print(f"Val labels:   {output_dir / 'val' / 'val.txt'}")
