"""Orchestrate training for run22_caption_compare using existing run21 config.

This wrapper creates a temporary config override (epochs=20, early_stopping disabled)
and invokes the existing `run_full_pipeline.py` per-category. It copies checkpoints
to `best.pt` and `last.pt` and collects basic metrics.
"""
import argparse
import os
import stat
import subprocess
import sys
import yaml
from pathlib import Path
import shutil
import json


def write_temp_config(base_cfg, out_path, overrides):
    cfg = dict(base_cfg)
    cfg.setdefault("training", {})
    cfg["training"].update(overrides)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _copy_tree(src_dir, dst_dir):
    if dst_dir.exists():
        def _onerror(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                raise

        shutil.rmtree(dst_dir, onerror=_onerror)
    shutil.copytree(src_dir, dst_dir)


def prepare_category_root(raw_root, split_json_path, dst_root, category):
    raw_root = Path(raw_root)
    dst_root = Path(dst_root)
    dst_cat = dst_root / category
    dst_cat.mkdir(parents=True, exist_ok=True)

    if category == "cable":
        src_cat = raw_root / "cable_resplit" / category
        if not src_cat.is_dir():
            raise FileNotFoundError(f"Cable resplit data not found: {src_cat}")
        _copy_tree(src_cat, dst_cat)
        return dst_cat

    if not Path(split_json_path).exists():
        raise FileNotFoundError(f"Split JSON not found: {split_json_path}")

    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Rebuild a category root with train/val/test subdirectories so the existing loader can consume it.
    for split_name in ["train", "val", "test"]:
        (dst_cat / split_name).mkdir(parents=True, exist_ok=True)
    (dst_cat / "ground_truth").mkdir(parents=True, exist_ok=True)

    src_cat = raw_root / category
    if not src_cat.is_dir():
        raise FileNotFoundError(f"Raw category data not found: {src_cat}")

    for split_name in ["train", "val", "test"]:
        for item in split_data.get(split_name, []):
            item_path = Path(os.path.normpath(item))
            defect_type = item_path.parent.name
            img_name = item_path.stem
            dst_img_dir = dst_cat / split_name / defect_type
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item_path, dst_img_dir / item_path.name)

            if defect_type != "good":
                src_mask_dir = src_cat / "ground_truth" / defect_type
                mask_src = None
                for mask_name in [f"{img_name}_mask.png", f"{img_name}.png"]:
                    candidate = src_mask_dir / mask_name
                    if candidate.exists():
                        mask_src = candidate
                        break
                if mask_src is not None:
                    dst_mask_dir = dst_cat / "ground_truth" / defect_type
                    dst_mask_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mask_src, dst_mask_dir / mask_src.name)

    for meta_name in ["readme.txt", "license.txt"]:
        src_meta = src_cat / meta_name
        if src_meta.exists():
            shutil.copy2(src_meta, dst_cat / meta_name)

    return dst_cat


def run_category(category, data_root, base_config, output_root, seed=42):
    cat_out = Path(output_root) / category
    cat_out.mkdir(parents=True, exist_ok=True)

    temp_cfg = cat_out / "config_run22.yaml"
    overrides = {
        "epochs": 20,
        "early_stopping": {"enabled": False, "patience": 4, "min_delta": 1e-4},
        "seed": seed,
    }
    # base_config may be a path or dict
    if isinstance(base_config, str):
        with open(base_config, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
    else:
        base = base_config

    # merge at training key if present; enforce epochs but preserve early_stopping from config
    base.setdefault("training", {})
    base_training = base["training"]
    base_training.update({"epochs": 20})
    write_temp_config(base, str(temp_cfg), {})

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_full_pipeline.py"),
        "--config",
        str(temp_cfg),
        "--data_root",
        str(data_root),
        "--categories",
        category,
        "--output_dir",
        str(cat_out),
        "--num_vis_samples",
        "5",
    ]

    print("Running training for category:", category)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {category}: {e}")
        return False

    # Copy/rename checkpoints for reproducible names
    best_src = None
    for candidate in cat_out.glob("**/best_model.pth"):
        best_src = candidate
        break
    if best_src:
        shutil.copy2(best_src, cat_out / "best.pt")

    # last checkpoint
    last_candidates = sorted(cat_out.glob("**/checkpoint_epoch*.pth"))
    if last_candidates:
        shutil.copy2(last_candidates[-1], cat_out / "last.pt")

    # Collect eval metrics if present
    eval_path = cat_out / "eval_results" / "results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Save a CSV summary
        csv_path = cat_out / "metrics.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            overall_img = results.get("overall", {}).get("image_level", {})
            for k, v in overall_img.items():
                f.write(f"image_{k},{v}\n")
            overall_pix = results.get("overall", {}).get("pixel_level", {})
            for k, v in overall_pix.items():
                f.write(f"pixel_{k},{v}\n")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="03_code/configs/default.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--categories", nargs="+", default=["cable", "transistor", "capsule"])
    parser.add_argument("--output_dir", type=str, default="runs/run22_caption_compare/model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load canonical default config then deep-merge any provided override
    def _deep_merge(a, b):
        for k, v in (b or {}).items():
            if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                _deep_merge(a[k], v)
            else:
                a[k] = v

    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    default_cfg_path = configs_dir / "default.yaml"
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    else:
        base_cfg = {}

    # If user provided an override config, load it and merge into base
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            override_cfg = yaml.safe_load(f) or {}
        _deep_merge(base_cfg, override_cfg)

    output_root = Path(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    run_root = output_root.parent if output_root.name == "model" else output_root
    split_root = run_root / "splits"
    if not split_root.exists():
        split_root = output_root / "splits"

    prepared_root = output_root / "prepared_data"
    prepared_root.mkdir(parents=True, exist_ok=True)

    for category in args.categories:
        split_json = split_root / f"{category}_split.json"
        prepare_category_root(args.data_root, split_json, prepared_root, category)

    for cat in args.categories:
        run_category(cat, str(prepared_root), base_cfg, str(output_root), seed=args.seed)

    print("Training orchestration complete. Outputs under:", output_root)


if __name__ == "__main__":
    main()
