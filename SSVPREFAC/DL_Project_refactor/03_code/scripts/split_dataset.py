"""Create deterministic 70/20/10 splits for specified categories and save indices.

Usage:
    python 03_code/scripts/split_dataset.py --data_root 04_data/datasets --categories transistor capsule --output_dir runs/run22_caption_compare/splits
"""
import argparse
import json
import os
from pathlib import Path
import random


def collect_samples_for_category(cat_dir):
    samples = []
    if not os.path.isdir(cat_dir):
        return samples

    # Walk known splits
    for split in ("train", "val", "test"):
        split_dir = os.path.join(cat_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for defect in sorted(os.listdir(split_dir)):
            defect_dir = os.path.join(split_dir, defect)
            if not os.path.isdir(defect_dir):
                continue
            for fn in sorted(os.listdir(defect_dir)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    samples.append({
                        "image_path": os.path.join(defect_dir, fn),
                        "defect_type": defect,
                        "label": 0 if defect.lower() == "good" else 1,
                    })

    # Also check top-level files (fallback)
    for fn in sorted(os.listdir(cat_dir)):
        p = os.path.join(cat_dir, fn)
        if os.path.isfile(p) and fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            samples.append({"image_path": p, "defect_type": "unknown", "label": 0})

    return samples


def split_preserve_label(samples, seed=42, ratios=(0.7, 0.2, 0.1)):
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(int(s.get("label", 0)), []).append(s)

    train, val, test = [], [], []
    for lbl, group in by_label.items():
        n = len(group)
        rng.shuffle(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # guarantee at least 1 in test if possible
        n_test = max(1, n - n_train - n_val) if n > 1 else 0
        # adjust if rounding caused mismatch
        if n_train + n_val + n_test > n:
            n_test = n - n_train - n_val

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:n_train + n_val + n_test])

    # If any leftover due to rounding, add to train
    assigned = len(train) + len(val) + len(test)
    if assigned < len(samples):
        extras = [s for s in samples if s not in train and s not in val and s not in test]
        train.extend(extras)

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--categories", nargs="+", default=["transistor", "capsule"])
    parser.add_argument("--output_dir", default="runs/run22_caption_compare/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        cat_dir = os.path.join(args.data_root, cat)
        print(f"Processing category: {cat} -> {cat_dir}")
        samples = collect_samples_for_category(cat_dir)
        if not samples:
            print(f"Warning: no samples found for {cat} at {cat_dir}")
            continue

        train, val, test = split_preserve_label(samples, seed=args.seed)

        summary = {
            "category": cat,
            "n_total": len(samples),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        }

        # Save indices as lists of image paths
        out_path = out / f"{cat}_split.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "train": [s["image_path"] for s in train],
                "val": [s["image_path"] for s in val],
                "test": [s["image_path"] for s in test],
            }, f, indent=2)

        print(f"Saved split for {cat} -> {out_path} ({len(train)}/{len(val)}/{len(test)})")


if __name__ == "__main__":
    main()
