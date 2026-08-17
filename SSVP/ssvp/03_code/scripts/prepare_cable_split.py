"""
Prepare randomized 70:20:10 train/test/val splits for one or more MVTec categories.

Input layout:
        04_data/datasets/<category>/
            train/good/
            test/good/
            test/<defect_type>/
            ground_truth/<defect_type>/

Some datasets in this repo use a nested wrapper directory, e.g.
        04_data/datasets/transistor/transistor/

Output layout:
        <output_root>/<category>/
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


def resolve_category_root(source_root: Path, category: str) -> Path:
    """Find the actual dataset root for a category, handling nested wrapper folders."""
    candidates = [
        source_root / category,
        source_root / category / category,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() or (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find train/test folders for category '{category}' under {source_root}"
    )


def main():
    parser = argparse.ArgumentParser(description="Resplit MVTec data into 70/20/10 train/test/val")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DATASETS_DIR),
        help="Source dataset root or category root parent",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DATASETS_DIR / "cable_resplit"),
        help="Root where resplit dataset will be saved",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cable"],
        help="Categories to resplit (default: cable)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source_root = Path(args.source)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source path not found: {source_root}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    combined_summary = {"seed": args.seed, "split": {"train": 0.7, "test": 0.2, "val": 0.1}, "categories": {}}

    for category in args.categories:
        category_source = resolve_category_root(source_root, category)
        category_output = output_root / category
        if category_output.exists():
            shutil.rmtree(category_output)

        pools = defaultdict(list)

        # Pool good samples from both original train and test/good.
        for origin in [category_source / "train" / "good", category_source / "test" / "good"]:
            if not origin.is_dir():
                continue
            for p in sorted(origin.iterdir()):
                if p.is_file() and is_image(p):
                    pools["good"].append((p, None))

        # Pool anomalous samples from original test defect folders.
        test_dir = category_source / "test"
        gt_dir = category_source / "ground_truth"
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

        category_summary = {"total": 0, "defects": {}}

        for defect, samples in sorted(pools.items()):
            rng.shuffle(samples)
            n_train, n_test, n_val = split_counts(len(samples))

            split_map = {
                "train": samples[:n_train],
                "test": samples[n_train:n_train + n_test],
                "val": samples[n_train + n_test:],
            }

            category_summary["defects"][defect] = {
                "total": len(samples),
                "train": len(split_map["train"]),
                "test": len(split_map["test"]),
                "val": len(split_map["val"]),
            }
            category_summary["total"] += len(samples)

            for split_name, split_samples in split_map.items():
                dest_img_dir = category_output / split_name / defect
                dest_img_dir.mkdir(parents=True, exist_ok=True)

                for i, (img_path, mask_path) in enumerate(split_samples):
                    # Add split index prefix to avoid filename collisions.
                    new_stem = f"{defect}_{split_name}_{i:05d}"
                    new_img = dest_img_dir / f"{new_stem}{img_path.suffix.lower()}"
                    shutil.copy2(img_path, new_img)

                    if defect != "good" and mask_path is not None:
                        mask_ext = mask_path.suffix.lower()
                        dest_mask_dir = category_output / "ground_truth" / defect
                        dest_mask_dir.mkdir(parents=True, exist_ok=True)
                        new_mask = dest_mask_dir / f"{new_stem}_mask{mask_ext}"
                        shutil.copy2(mask_path, new_mask)

        summary_path = category_output / "split_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Seed: {args.seed}\n")
            f.write("Split ratio: train=0.7, test=0.2, val=0.1\n\n")
            for defect, counts in category_summary["defects"].items():
                f.write(
                    f"{defect}: total={counts['total']}, train={counts['train']}, "
                    f"test={counts['test']}, val={counts['val']}\n"
                )

        combined_summary["categories"][category] = {
            "source": str(category_source),
            "output": str(category_output),
            **category_summary,
        }

        print(f"Resplit dataset saved to: {category_output}")
        print(f"Summary written to: {summary_path}")

    combined_summary_path = output_root / "split_summary.txt"
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        f.write(f"Seed: {combined_summary['seed']}\n")
        f.write("Split ratio: train=0.7, test=0.2, val=0.1\n\n")
        for category, info in combined_summary["categories"].items():
            f.write(f"[{category}] source={info['source']} output={info['output']} total={info['total']}\n")
            for defect, counts in info["defects"].items():
                f.write(
                    f"  {defect}: total={counts['total']}, train={counts['train']}, "
                    f"test={counts['test']}, val={counts['val']}\n"
                )
            f.write("\n")

    print(f"Combined summary written to: {combined_summary_path}")


if __name__ == "__main__":
    main()
