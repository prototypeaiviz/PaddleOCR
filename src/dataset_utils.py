"""Utilities for rendering a toy OCR dataset and saving PaddleOCR label files."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_CANDIDATES = [
    # Common Ubuntu / Linux font locations.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def load_words(word_file: Path) -> List[str]:
    """Load candidate labels from a text file."""
    words = [line.strip() for line in word_file.read_text(encoding="utf-8").splitlines()]
    words = [w for w in words if w]
    if not words:
        raise ValueError(f"No words found in: {word_file}")
    return words


def choose_font(font_size: int = 24) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a real font; fall back to PIL's default font if needed."""
    for font_path in DEFAULT_FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def render_word_image(
    text: str,
    image_height: int,
    min_width: int,
    max_width: int,
    background_range: Tuple[int, int] = (220, 255),
    foreground_range: Tuple[int, int] = (0, 40),
):
    """Render a single text label into a synthetic image.

    The function creates a light background image, draws dark text, and returns
    a PIL image. Width is chosen dynamically so that short words are not too
    squashed and longer words still fit.
    """
    # Create a temporary font and estimate text size first.
    font = choose_font(font_size=max(16, image_height - 8))

    # PIL sizing helper.
    temp_img = Image.new("RGB", (max_width, image_height), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    # Choose a width with some padding but within the requested bounds.
    width = max(min_width, min(max_width, text_w + 20))

    bg = random.randint(*background_range)
    fg = random.randint(*foreground_range)
    image = Image.new("RGB", (width, image_height), (bg, bg, bg))
    draw = ImageDraw.Draw(image)

    # Center the text inside the image.
    x = max(5, (width - text_w) // 2)
    y = max(0, (image_height - text_h) // 2 - bbox[1])
    draw.text((x, y), text, font=font, fill=(fg, fg, fg))

    return image


def write_label_file(label_path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    """Write PaddleOCR-style label file: relative_path<TAB>label"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for rel_path, label in rows:
            f.write(f"{rel_path}\t{label}\n")
