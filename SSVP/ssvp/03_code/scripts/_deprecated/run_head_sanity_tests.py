"""
Run a 3-epoch head-only sanity suite for pruning/LoRA variants.

Variants:
  1) width-wise pruning (heads only)
  2) depth-wise pruning (heads only)
  3) differentiated pruning (heads only)
  4) LoRA rank-2 (heads only, no backbone)

Usage:
    python 03_code/scripts/run_head_sanity_tests.py \
            --config 03_code/configs/default.yaml \
            --data_root 04_data/datasets/cable_resplit/cable \
            --output_root 05_results/ablations/head_sanity_3ep \
      --epochs 3
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from path_utils import DATASETS_DIR, LOGS_DIR, SCRIPTS_DIR, default_config_path


TRAIN_SCRIPT = str(SCRIPTS_DIR / "train.py")


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value


def run_cmd(args):
    print("Running:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_training_status(run_dir):
    summary_path = run_dir / "training_summary.json"
    metrics_path = run_dir / "training_metrics.json"

    status = {
        "run_dir": str(run_dir),
        "has_best_checkpoint": (run_dir / "best_model.pth").exists(),
        "has_training_summary": summary_path.exists(),
        "has_training_metrics": metrics_path.exists(),
    }

    if summary_path.exists():
        training_summary = load_json(summary_path)
        status["best_val_loss"] = training_summary.get("best_val_loss")
        status["epochs_completed"] = training_summary.get("epochs_completed")

    return status


def main():
    parser = argparse.ArgumentParser(description="Run 3-epoch head-only sanity variants")
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--data_root", type=str, default=str(DATASETS_DIR / "cable_resplit" / "cable"))
    parser.add_argument("--output_root", type=str, default=str(LOGS_DIR / "head_sanity_3ep"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    variants = [
        {
            "name": "width_pruning_heads",
            "overrides": {
                "head_pruning": {
                    "enabled": True,
                    "mode": "width",
                    "width_keep_ratio": 0.75,
                },
                "lora": {
                    "enabled": False,
                },
            },
        },
        {
            "name": "depth_pruning_heads",
            "overrides": {
                "head_pruning": {
                    "enabled": True,
                    "mode": "depth",
                    "depth_keep_ratio": 0.75,
                },
                "lora": {
                    "enabled": False,
                },
            },
        },
        {
            "name": "differentiated_pruning_heads",
            "overrides": {
                "head_pruning": {
                    "enabled": True,
                    "mode": "differentiated",
                    "module_keep_ratios": {
                        "hsvs": 0.70,
                        "vcpg": 0.85,
                        "vtam": 0.65,
                    },
                    "depth_keep_ratio": 0.75,
                },
                "lora": {
                    "enabled": False,
                },
            },
        },
        {
            "name": "lora_rank2_heads_only",
            "overrides": {
                "head_pruning": {
                    "enabled": False,
                },
                "lora": {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 8,
                    "dropout": 0.05,
                    "freeze_base": True,
                    "scopes": ["hsvs", "vcpg", "vtam"],
                    "head_target_substrings": [],
                },
            },
        },
    ]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_config": args.config,
        "data_root": args.data_root,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "variants": {},
    }

    summary_path = output_root / "sanity_summary.json"

    for variant in variants:
        name = variant["name"]
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        temp_cfg = copy.deepcopy(base_cfg)
        deep_update(temp_cfg, variant["overrides"])
        temp_cfg.setdefault("training", {})
        temp_cfg["training"]["epochs"] = int(args.epochs)

        cfg_path = run_dir / "_sanity_config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(temp_cfg, f, sort_keys=False)

        train_summary_path = run_dir / "training_summary.json"
        if args.skip_existing and train_summary_path.exists():
            print(f"Skipping existing run: {name}", flush=True)
            status = collect_training_status(run_dir)
            status["run_status"] = "skipped_existing"
            summary["variants"][name] = status
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            continue

        cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--config",
            str(cfg_path),
            "--output_dir",
            str(run_dir),
            "--seed",
            str(args.seed),
        ]

        if args.data_root:
            dr = Path(args.data_root)
            if (dr / "train").is_dir() or (dr / "test").is_dir():
                cmd += ["--data_root", str(dr.parent)]
                cmd += ["--categories", dr.name]
            else:
                cmd += ["--data_root", str(dr)]

        try:
            run_cmd(cmd)
            status = collect_training_status(run_dir)
            status["run_status"] = "ok"
        except subprocess.CalledProcessError as exc:
            status = collect_training_status(run_dir)
            status["run_status"] = "failed"
            status["return_code"] = int(exc.returncode)

        summary["variants"][name] = status
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\nSanity summary written to:", summary_path)


if __name__ == "__main__":
    main()
