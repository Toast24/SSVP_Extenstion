"""
MVTec-AD Dataset Loader for SSVP.
"""

import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

from .transforms import AugmentedTransform, DualResizeTransform


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}
OBJECT_CATEGORIES = set(MVTEC_CATEGORIES) - TEXTURE_CATEGORIES


class MVTecDataset(Dataset):
    """
    MVTec-style dataset.

    Supported split layouts under each category:
      - train/
      - val/ (optional)
      - test/
      - ground_truth/

    By default, `split=train` loads only `train/good` to preserve baseline behavior.
    Set `train_all_types=True` to load all defect folders from `train/`.
    """

    def __init__(
        self,
        data_root,
        categories=None,
        split="train",
        img_size=518,
        mask_size=37,
        augment=False,
        train_all_types=False,
        augment_config=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.categories = categories or MVTEC_CATEGORIES
        self.train_all_types = bool(train_all_types)

        if augment and split == "train":
            self.transform = AugmentedTransform(img_size, mask_size, augment_config=augment_config)
        else:
            self.transform = DualResizeTransform(img_size, mask_size)

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for category in self.categories:
            cat_dir = os.path.join(self.data_root, category)
            if not os.path.isdir(cat_dir):
                print(f"Warning: Category directory not found: {cat_dir}")
                continue

            if self.split == "train" and not self.train_all_types:
                good_dir = os.path.join(cat_dir, "train", "good")
                if os.path.isdir(good_dir):
                    for img_path in sorted(glob.glob(os.path.join(good_dir, "*.*"))):
                        if self._is_image(img_path):
                            self.samples.append(
                                {
                                    "image_path": img_path,
                                    "mask_path": None,
                                    "label": 0,
                                    "category": category,
                                    "defect_type": "good",
                                }
                            )
                continue

            if self.split in {"train", "val", "test"}:
                self._load_split_all_types(cat_dir, category, self.split)
                continue

            if self.split == "all":
                for sub_split in ["train", "val", "test"]:
                    self._load_split_all_types(cat_dir, category, sub_split)

    def _load_split_all_types(self, cat_dir, category, split_name):
        split_dir = os.path.join(cat_dir, split_name)
        gt_dir = os.path.join(cat_dir, "ground_truth")
        if not os.path.isdir(split_dir):
            return

        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = os.path.join(split_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue

            for img_path in sorted(glob.glob(os.path.join(defect_dir, "*.*"))):
                if not self._is_image(img_path):
                    continue

                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if defect_type == "good":
                    mask_path = None
                    label = 0
                else:
                    mask_path = self._find_mask(os.path.join(gt_dir, defect_type), img_name)
                    label = 1

                self.samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "label": label,
                        "category": category,
                        "defect_type": defect_type,
                    }
                )

    @staticmethod
    def _is_image(path):
        return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))

    @staticmethod
    def _find_mask(mask_dir, img_name):
        if not os.path.isdir(mask_dir):
            return None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            mask_path = os.path.join(mask_dir, img_name + "_mask" + ext)
            if os.path.exists(mask_path):
                return mask_path
            mask_path = os.path.join(mask_dir, img_name + ext)
            if os.path.exists(mask_path):
                return mask_path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if sample["mask_path"] is not None and os.path.exists(sample["mask_path"]):
            mask = Image.open(sample["mask_path"]).convert("L")
        else:
            mask = None

        img_tensor, mask_tensor, mask_full = self.transform(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "mask_full": mask_full,
            "label": sample["label"],
            "category": sample["category"],
            "defect_type": sample["defect_type"],
            "image_path": sample["image_path"],
        }


def get_mvtec_dataloaders(config, source_categories=None, target_categories=None):
    data_cfg = config["data"]
    train_cfg = config["training"]

    use_anomaly_supervision = bool(train_cfg.get("use_anomaly_supervision", False))
    augment_cfg = train_cfg.get("augment", {})
    augment_enabled = bool(augment_cfg.get("enabled", False))
    train_split = "all" if use_anomaly_supervision else "train"

    train_dataset = MVTecDataset(
        data_root=data_cfg["data_root"],
        categories=source_categories,
        split=train_split,
        img_size=data_cfg["img_size"],
        mask_size=data_cfg["mask_size"],
        augment=augment_enabled,
        train_all_types=use_anomaly_supervision,
        augment_config=augment_cfg,
    )

    test_dataset = MVTecDataset(
        data_root=data_cfg["data_root"],
        categories=target_categories,
        split="test",
        img_size=data_cfg["img_size"],
        mask_size=data_cfg["mask_size"],
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    return train_loader, test_loader


def get_mvtec_train_val_dataloaders(config, source_categories=None, val_ratio=None, seed=42):
    """
    Create train/validation loaders.

    If `training.use_explicit_val_split` is true, uses dataset `train/` and `val/`.
    Otherwise falls back to splitting the train dataset by `training.val_split`.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    use_explicit_val = bool(train_cfg.get("use_explicit_val_split", False))
    train_all_types = bool(train_cfg.get("train_all_types", False))
    augment_cfg = train_cfg.get("augment", {})
    augment_enabled = bool(augment_cfg.get("enabled", False))

    if use_explicit_val:
        train_dataset = MVTecDataset(
            data_root=data_cfg["data_root"],
            categories=source_categories,
            split="train",
            img_size=data_cfg["img_size"],
            mask_size=data_cfg["mask_size"],
            augment=augment_enabled,
            train_all_types=train_all_types,
            augment_config=augment_cfg,
        )
        val_dataset = MVTecDataset(
            data_root=data_cfg["data_root"],
            categories=source_categories,
            split="val",
            img_size=data_cfg["img_size"],
            mask_size=data_cfg["mask_size"],
            augment=False,
            train_all_types=False,
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise RuntimeError(
                "Explicit train/val split requested but train or val data is empty."
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
            pin_memory=train_cfg["pin_memory"],
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            pin_memory=train_cfg["pin_memory"],
            drop_last=False,
        )
        return train_loader, val_loader

    full_train_dataset = MVTecDataset(
        data_root=data_cfg["data_root"],
        categories=source_categories,
        split="train",
        img_size=data_cfg["img_size"],
        mask_size=data_cfg["mask_size"],
        augment=augment_enabled,
        train_all_types=train_all_types,
        augment_config=augment_cfg,
    )

    ratio = train_cfg.get("val_split", 0.2) if val_ratio is None else float(val_ratio)
    ratio = min(max(ratio, 0.05), 0.5)

    total_samples = len(full_train_dataset)
    if total_samples < 2:
        raise RuntimeError("Not enough train samples to build train/val split.")

    val_len = max(1, int(total_samples * ratio))
    train_len = total_samples - val_len
    if train_len < 1:
        train_len = 1
        val_len = total_samples - 1

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=generator,
    )

    val_dataset = MVTecDataset(
        data_root=data_cfg["data_root"],
        categories=source_categories,
        split="train",
        img_size=data_cfg["img_size"],
        mask_size=data_cfg["mask_size"],
        augment=False,
        train_all_types=train_all_types,
    )
    val_subset.dataset = val_dataset

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
    )

    return train_loader, val_loader
