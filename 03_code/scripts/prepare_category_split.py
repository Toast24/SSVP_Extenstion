"""
Prepare randomized 70:20:10 train/test/val splits for the cable dataset.

Input layout:
    04_data/datasets/cable/
      train/good/
      test/good/
      test/<defect_type>/
      ground_truth/<defect_type>/

Output layout:
    04_data/datasets/cable_resplit/cable/
      train/<defect_type>/
      test/<defect_type>/
      val/<defect_type>/
      ground_truth/<defect_type>/

For anomalous images, matching masks are copied to ground_truth with renamed stems so
MVTec mask lookup remains valid.
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from path_utils import DATASETS_DIR


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def find_mask(mask_dir: Path, image_stem: str):
    if not mask_dir.is_dir():
        return None
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        cand = mask_dir / f"{image_stem}_mask{ext}"
        if cand.exists():
            return cand
        cand = mask_dir / f"{image_stem}{ext}"
        if cand.exists():
            return cand
    return None


def split_counts(n: int):
    n_train = int(round(n * 0.7))
    n_test = int(round(n * 0.2))
    n_val = n - n_train - n_test

    if n >= 3:
        # Ensure each split has at least one sample when feasible.
        if n_train == 0:
            n_train = 1
        if n_test == 0:
            n_test = 1
        n_val = n - n_train - n_test
        if n_val <= 0:
            n_val = 1
            if n_train > n_test:
                n_train -= 1
            else:
                n_test -= 1

    return n_train, n_test, n_val


def copy_sample(image_path: Path, dest_img_dir: Path):
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest_img_dir / image_path.name)


def main():
    parser = argparse.ArgumentParser(description="Resplit cable data into 70/20/10 train/test/val")
    parser.add_argument("--source", type=str,
                        default=str(DATASETS_DIR / "cable"),
                        help="Source dataset directory (e.g., 04_data/datasets/cable)")
    parser.add_argument("--category", type=str, default="cable",
                        help="MVTec AD category name (e.g., cable, bottle, leather)")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DATASETS_DIR / "cable_resplit"),
        help="Root where resplit dataset will be saved",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source = Path(args.source)
    if not source.is_dir():
        raise FileNotFoundError(f"Source path not found: {source}")

    out_category = Path(args.output_root) / args.category
    if out_category.exists():
        shutil.rmtree(out_category)

    pools = defaultdict(list)

    # Pool good samples from both original train and test/good.
    for origin in [source / "train" / "good", source / "test" / "good"]:
        if not origin.is_dir():
            continue
        for p in sorted(origin.iterdir()):
            if p.is_file() and is_image(p):
                pools["good"].append((p, None))

    # Pool anomalous samples from original test defect folders.
    test_dir = source / "test"
    gt_dir = source / "ground_truth"
    if test_dir.is_dir():
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue
            defect = defect_dir.name
            for img in sorted(defect_dir.iterdir()):
                if not img.is_file() or not is_image(img):
                    continue
                mask = find_mask(gt_dir / defect, img.stem)
                pools[defect].append((img, mask))

    summary = {"seed": args.seed, "split": {"train": 0.7, "test": 0.2, "val": 0.1}, "defects": {}}

    for defect, samples in sorted(pools.items()):
        rng.shuffle(samples)
        n_train, n_test, n_val = split_counts(len(samples))

        split_map = {
            "train": samples[:n_train],
            "test": samples[n_train:n_train + n_test],
            "val": samples[n_train + n_test:],
        }

        summary["defects"][defect] = {
            "total": len(samples),
            "train": len(split_map["train"]),
            "test": len(split_map["test"]),
            "val": len(split_map["val"]),
        }

        for split_name, split_samples in split_map.items():
            dest_img_dir = out_category / split_name / defect
            dest_img_dir.mkdir(parents=True, exist_ok=True)

            for i, (img_path, mask_path) in enumerate(split_samples):
                # Add split index prefix to avoid filename collisions.
                new_stem = f"{defect}_{split_name}_{i:05d}"
                new_img = dest_img_dir / f"{new_stem}{img_path.suffix.lower()}"
                shutil.copy2(img_path, new_img)

                if defect != "good" and mask_path is not None:
                    mask_ext = mask_path.suffix.lower()
                    dest_mask_dir = out_category / "ground_truth" / defect
                    dest_mask_dir.mkdir(parents=True, exist_ok=True)
                    new_mask = dest_mask_dir / f"{new_stem}_mask{mask_ext}"
                    shutil.copy2(mask_path, new_mask)

    summary_path = out_category / "split_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Seed: {summary['seed']}\n")
        f.write("Split ratio: train=0.7, test=0.2, val=0.1\n\n")
        for defect, counts in summary["defects"].items():
            f.write(
                f"{defect}: total={counts['total']}, train={counts['train']}, "
                f"test={counts['test']}, val={counts['val']}\n"
            )

    print(f"Resplit dataset saved to: {out_category}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
