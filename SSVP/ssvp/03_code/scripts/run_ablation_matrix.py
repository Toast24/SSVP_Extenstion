"""
Run a compact robustness ablation matrix plus a LoRA experiment.

This script creates per-variant temp configs, runs the full pipeline, and writes
an aggregated summary for clean/noisy metrics and clean->noisy deltas.

Usage:
    python 03_code/scripts/run_ablation_matrix.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/ablation_m1 --epochs 3
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from path_utils import (
    ABLATION_JSON_DIR,
    ABLATIONS_DIR,
    SCRIPTS_DIR,
    default_config_path,
)


RUN_FULL_PIPELINE_SCRIPT = str(SCRIPTS_DIR / "run_full_pipeline.py")


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v


def run_cmd(args):
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_variant_metrics(run_dir: Path):
    clean_path = run_dir / "eval_results" / "results.json"
    noisy_path = run_dir / "noise_eval" / "noise_results.json"
    report_path = run_dir / "noise_eval" / "robustness_report.json"

    clean = load_json(clean_path)["overall"]
    noisy = load_json(noisy_path)["overall"]
    report = load_json(report_path).get("delta", {})

    return {
        "clean": {
            "i_auroc": clean["image_level"]["auroc"],
            "i_f1": clean["image_level"]["f1_max"],
            "i_ap": clean["image_level"]["ap"],
            "p_auroc": clean["pixel_level"]["auroc"],
            "p_pro": clean["pixel_level"]["pro"],
            "p_ap": clean["pixel_level"]["ap"],
        },
        "noisy": {
            "i_auroc": noisy["image_level"]["auroc"],
            "i_f1": noisy["image_level"]["f1_max"],
            "i_ap": noisy["image_level"]["ap"],
            "p_auroc": noisy["pixel_level"]["auroc"],
            "p_pro": noisy["pixel_level"]["pro"],
            "p_ap": noisy["pixel_level"]["ap"],
        },
        "delta": {
            "i_auroc": report.get("image_auroc"),
            "i_f1": report.get("image_f1_max"),
            "i_ap": report.get("image_ap"),
            "p_auroc": report.get("pixel_auroc"),
            "p_pro": report.get("pixel_pro"),
            "p_ap": report.get("pixel_ap"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run robustness ablations + LoRA experiment")
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=str(ABLATIONS_DIR / "ablation_matrix"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_vis_samples", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional subset of variant names to run",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    variants = [
        {
            "name": "no_consistency",
            "overrides": {
                "training": {
                    "consistency": {"enabled": False, "lambda": 0.0},
                }
            },
        },
        {
            "name": "no_tta",
            "overrides": {
                "eval": {
                    "tta": {"enabled": False, "hflip": False},
                }
            },
        },
        {
            "name": "no_robust_postproc",
            "overrides": {
                "eval": {
                    "clip_percentiles": None,
                    "median_ksize": 0,
                }
            },
        },
        {
            "name": "weak_aug",
            "overrides": {
                "training": {
                    "augment": {
                        "hflip": True,
                        "color_jitter": 0.1,
                        "sharpness_p": 0.0,
                        "blur_p": 0.0,
                        "blur_sigma": [0.1, 0.1],
                        "noise_std": 0.0,
                        "noise_p": 0.0,
                    }
                }
            },
        },
        {
            "name": "lora_no_consistency",
            "overrides": {
                "training": {
                    "consistency": {"enabled": False, "lambda": 0.0},
                },
                "lora": {
                    "enabled": True,
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "freeze_base": True,
                    "scopes": ["hsvs", "vcpg", "vtam"],
                    "target_substrings": [],
                },
            },
        },
    ]

    if args.variants:
        requested = set(args.variants)
        variants = [v for v in variants if v["name"] in requested]
        if not variants:
            raise ValueError("No matching variants found for --variants")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_config": args.config,
        "epochs": args.epochs,
        "variants": {},
    }

    for variant in variants:
        name = variant["name"]
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        temp_cfg = copy.deepcopy(base_cfg)
        deep_update(temp_cfg, variant["overrides"])
        temp_cfg["training"]["epochs"] = int(args.epochs)

        temp_cfg_path = run_dir / "_ablation_config.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(temp_cfg, f, sort_keys=False)

        clean_results = run_dir / "eval_results" / "results.json"
        noisy_results = run_dir / "noise_eval" / "noise_results.json"
        if args.skip_existing and clean_results.exists() and noisy_results.exists():
            print(f"Skipping existing variant: {name}")
        else:
            cmd = [
                sys.executable,
                RUN_FULL_PIPELINE_SCRIPT,
                "--config",
                str(temp_cfg_path),
                "--output_dir",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--num_vis_samples",
                str(args.num_vis_samples),
            ]
            if args.data_root:
                cmd += ["--data_root", args.data_root]
            run_cmd(cmd)

        summary["variants"][name] = collect_variant_metrics(run_dir)

        summary_path = output_root / "ablation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        ABLATION_JSON_DIR.mkdir(parents=True, exist_ok=True)
        with open(ABLATION_JSON_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\nAblation summary written to:", output_root / "ablation_summary.json")


if __name__ == "__main__":
    main()
