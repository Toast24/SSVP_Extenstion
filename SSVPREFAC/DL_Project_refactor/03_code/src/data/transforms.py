"""
Dual-Resolution Image Transforms for SSVP.

Both CLIP ViT-L/14 and DINOv2 ViT-L/14 use patch_size=14.
Input resolution: 518×518 → spatial grid: 37×37 (518/14 = 37).

Masks are resized to the feature grid size (37×37) for loss computation.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


class AdditiveGaussianNoise:
    """Apply additive Gaussian noise on tensor image after normalization."""

    def __init__(self, std=0.05, p=0.5):
        self.std = float(std)
        self.p = float(p)

    def __call__(self, img_tensor):
        if torch.rand(1).item() > self.p or self.std <= 0:
            return img_tensor
        noise = torch.randn_like(img_tensor) * self.std
        return torch.clamp(img_tensor + noise, -5.0, 5.0)


class DualResizeTransform:
    """
    Produces two versions of each image (for CLIP and DINO backbones)
    and resizes the mask to match the feature grid.

    Since both backbones use patch_size=14, both images are resized to
    the same resolution (518×518), so we produce a single image.

    Args:
        img_size:  Input image resolution (518)
        mask_size: Feature grid size for mask (37)
    """

    def __init__(self, img_size=518, mask_size=37):
        self.img_size = img_size
        self.mask_size = mask_size

        # Image transforms (both backbones use the same resolution)
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Mask transform (resize to feature grid size)
        self.mask_transform = T.Compose([
            T.Resize((mask_size, mask_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        # Full-resolution mask (for evaluation)
        self.mask_full_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __call__(self, image, mask=None):
        """
        Args:
            image: PIL Image
            mask:  PIL Image (grayscale, binary) or None

        Returns:
            img_tensor:      [3, img_size, img_size]
            mask_tensor:     [1, mask_size, mask_size] ∈ {0, 1}
            mask_full:       [1, img_size, img_size] ∈ {0, 1} (for eval)
        """
        img_tensor = self.img_transform(image)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            mask_tensor = self.mask_transform(mask)
            mask_full = self.mask_full_transform(mask)

            # Binarize mask (threshold at 0.5)
            mask_tensor = (mask_tensor > 0.5).float()
            mask_full = (mask_full > 0.5).float()
        else:
            mask_tensor = torch.zeros(1, self.mask_size, self.mask_size)
            mask_full = torch.zeros(1, self.img_size, self.img_size)

        return img_tensor, mask_tensor, mask_full


class AugmentedTransform(DualResizeTransform):
    """
    Extends DualResizeTransform with training-time augmentations.

    Only applies augmentations that are safe for anomaly detection
    (no geometric transforms that would misalign masks).
    """

    def __init__(self, img_size=518, mask_size=37, augment_config=None):
        super().__init__(img_size, mask_size)

        augment_config = augment_config or {}
        use_hflip = bool(augment_config.get("hflip", True))
        color_jitter = float(augment_config.get("color_jitter", 0.2))
        sharpness_p = float(augment_config.get("sharpness_p", 0.2))
        blur_p = float(augment_config.get("blur_p", 0.3))
        blur_sigma = augment_config.get("blur_sigma", [0.1, 1.5])
        if isinstance(blur_sigma, (int, float)):
            blur_sigma = [float(blur_sigma), float(blur_sigma)]
        noise_std = float(augment_config.get("noise_std", 0.07))
        noise_p = float(augment_config.get("noise_p", 0.5))

        self.aug_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5 if use_hflip else 0.0),
            T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=max(color_jitter * 0.5, 0.0),
                hue=min(max(color_jitter * 0.1, 0.0), 0.5),
            ),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=sharpness_p),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=tuple(blur_sigma))], p=blur_p),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            AdditiveGaussianNoise(std=noise_std, p=noise_p),
        ])

    def __call__(self, image, mask=None):
        """Apply augmented transforms for training."""
        # Use augmented transform for image
        img_tensor = self.aug_transform(image)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            mask_tensor = self.mask_transform(mask)
            mask_full = self.mask_full_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
            mask_full = (mask_full > 0.5).float()
        else:
            mask_tensor = torch.zeros(1, self.mask_size, self.mask_size)
            mask_full = torch.zeros(1, self.img_size, self.img_size)

        return img_tensor, mask_tensor, mask_full
