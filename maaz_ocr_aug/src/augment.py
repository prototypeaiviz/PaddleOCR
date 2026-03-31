"""
OCR-specific augmentation module.

This file is intentionally simple and heavily commented so you can study it.
The main idea is: change how the text image looks, but keep the label the same.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    """Configuration container for the custom OCR augmenter.

    These probabilities control *whether* each transform is applied.
    Keep them mild at first. Strong augmentation can make the text unreadable.
    """

    p_perspective: float = 0.25
    p_rotation: float = 0.25
    p_blur: float = 0.20
    p_noise: float = 0.20
    p_brightness_contrast: float = 0.20
    p_horizontal_stretch: float = 0.15

    # Strength values for the transforms.
    max_rotation_degrees: float = 5.0
    perspective_distortion: float = 0.08
    stretch_min_scale: float = 0.90
    stretch_max_scale: float = 1.10
    noise_sigma_min: float = 5.0
    noise_sigma_max: float = 15.0
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    brightness_min: float = -20.0
    brightness_max: float = 20.0


class OCRAugmentor:
    """Apply OCR-friendly image augmentation.

    Why this exists:
    ----------------
    OCR recognition models often fail when text is blurry, slightly rotated,
    stretched, noisy, or photographed at an angle. This class simulates those
    problems during training.

    Important rule:
    ---------------
    The *image* changes, but the *label string* must remain exactly the same.
    """

    def __init__(self, config: AugmentConfig | None = None) -> None:
        self.cfg = config or AugmentConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run augmentation on an image.

        Args:
            image: H x W x C uint8 image in BGR format.

        Returns:
            Augmented image in the same uint8 BGR format.
        """
        img = image.copy()

        # We apply the transforms in a reasonable order.
        # Geometry first, then visual degradation.
        if random.random() < self.cfg.p_perspective:
            img = self.random_perspective(img)

        if random.random() < self.cfg.p_rotation:
            img = self.random_rotation(img)

        if random.random() < self.cfg.p_horizontal_stretch:
            img = self.horizontal_stretch(img)

        if random.random() < self.cfg.p_blur:
            img = self.gaussian_blur(img)

        if random.random() < self.cfg.p_noise:
            img = self.add_gaussian_noise(img)

        if random.random() < self.cfg.p_brightness_contrast:
            img = self.adjust_brightness_contrast(img)

        return img

    def random_rotation(self, img: np.ndarray) -> np.ndarray:
        """Rotate the image by a small random angle.

        Small rotations are common in real OCR images due to camera angle or
        imperfect cropping. We keep the angle small so the text stays readable.
        """
        h, w = img.shape[:2]
        angle = random.uniform(-self.cfg.max_rotation_degrees, self.cfg.max_rotation_degrees)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def random_perspective(self, img: np.ndarray) -> np.ndarray:
        """Apply a mild perspective warp.

        This simulates text photographed at an angle rather than perfectly
        front-facing.
        """
        h, w = img.shape[:2]
        distortion = self.cfg.perspective_distortion

        src = np.float32(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ]
        )

        max_dx = max(1, int(w * distortion))
        max_dy = max(1, int(h * distortion))

        dst = np.float32(
            [
                [random.randint(0, max_dx), random.randint(0, max_dy)],
                [w - 1 - random.randint(0, max_dx), random.randint(0, max_dy)],
                [w - 1 - random.randint(0, max_dx), h - 1 - random.randint(0, max_dy)],
                [random.randint(0, max_dx), h - 1 - random.randint(0, max_dy)],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    def horizontal_stretch(self, img: np.ndarray) -> np.ndarray:
        """Stretch or squeeze the image horizontally.

        Why do this?
        ------------
        Real OCR crops can have slightly different aspect ratios or spacing.
        Mild horizontal stretch helps the model tolerate that variation.
        """
        h, w = img.shape[:2]
        scale = random.uniform(self.cfg.stretch_min_scale, self.cfg.stretch_max_scale)
        new_w = max(1, int(w * scale))

        stretched = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)

        # Resize back to the original width so the downstream pipeline has a
        # consistent image shape.
        restored = cv2.resize(stretched, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored

    def gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply mild blur.

        This simulates soft focus, motion blur, or low-quality screenshots.
        """
        kernel = random.choice([3, 5])
        return cv2.GaussianBlur(img, (kernel, kernel), 0)

    def add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise.

        This simulates sensor noise or compression artifacts.
        """
        sigma = random.uniform(self.cfg.noise_sigma_min, self.cfg.noise_sigma_max)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def adjust_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Change brightness and contrast.

        This simulates lighting changes, scanner variation, and display/camera
        differences.
        """
        alpha = random.uniform(self.cfg.contrast_min, self.cfg.contrast_max)
        beta = random.uniform(self.cfg.brightness_min, self.cfg.brightness_max)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
