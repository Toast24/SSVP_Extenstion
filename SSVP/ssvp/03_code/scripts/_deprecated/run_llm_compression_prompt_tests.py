"""
Run fast caption-side LLM experiments on an existing checkpoint.

Variants:
  - baseline: default caption generation
  - llm_compression: INT8 dynamic quantization on caption text transformer
  - prompt_improvement: domain fine-tuning for caption generation prompts
  - combined: both quantization and domain fine-tuning

This script uses run_full_pipeline.py in skip-train + skip-eval mode to keep tests fast.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from path_utils import DATASETS_DIR, LOGS_DIR, SCRIPTS_DIR, default_config_path


VARIANT_FLAGS = {
    "baseline": [],
    "llm_compression": ["--caption_text_int8"],
    "prompt_improvement": ["--caption_finetune"],
    "combined": ["--caption_text_int8", "--caption_finetune"],
}


RUN_FULL_PIPELINE_SCRIPT = str(SCRIPTS_DIR / "run_full_pipeline.py")


def run_cmd(args):
    print("Running:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def load_json_if_exists(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_variant(python_exe, variant_name, flags, args):
    run_dir = Path(args.output_root) / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        RUN_FULL_PIPELINE_SCRIPT,
        "--config",
        args.config,
        "--output_dir",
        str(run_dir),
        "--skip_train",
        "--skip_eval",
        "--checkpoint",
        args.checkpoint,
        "--num_vis_samples",
        str(args.num_vis_samples),
    ]

    if args.data_root:
        cmd += ["--data_root", args.data_root]

    cmd += flags

    t0 = time.perf_counter()
    run_cmd(cmd)
    elapsed_sec = time.perf_counter() - t0

    vis_dir = run_dir / "visualizations_with_captions"
    png_count = len(list(vis_dir.glob("*.png"))) if vis_dir.exists() else 0
    txt_count = len(list(vis_dir.glob("*.txt"))) if vis_dir.exists() else 0

    finetune_summary = load_json_if_exists(run_dir / "caption_finetune_summary.json")

    return {
        "variant": variant_name,
        "flags": flags,
        "run_dir": str(run_dir),
        "elapsed_sec": round(elapsed_sec, 3),
        "visualizations_png": int(png_count),
        "captions_txt": int(txt_count),
        "caption_finetune_summary": finetune_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Run LLM compression and prompt improvement caption tests")
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--checkpoint", type=str, default=str(LOGS_DIR / "run21_resplit_15es" / "best_model.pth"))
    parser.add_argument("--data_root", type=str, default=str(DATASETS_DIR / "cable_resplit" / "cable"))
    parser.add_argument("--output_root", type=str, default=str(LOGS_DIR / "llm_prompt_tests"))
    parser.add_argument("--num_vis_samples", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "llm_compression", "prompt_improvement", "combined"],
        help="Subset of variants to run",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requested = []
    for name in args.variants:
        if name not in VARIANT_FLAGS:
            raise ValueError(f"Unknown variant '{name}'. Available: {sorted(VARIANT_FLAGS.keys())}")
        requested.append(name)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "num_vis_samples": int(args.num_vis_samples),
        "variants": {},
    }

    python_exe = sys.executable
    for variant_name in requested:
        result = run_variant(
            python_exe=python_exe,
            variant_name=variant_name,
            flags=VARIANT_FLAGS[variant_name],
            args=args,
        )
        summary["variants"][variant_name] = result

        summary_path = output_root / "llm_prompt_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    summary_path = output_root / "llm_prompt_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nLLM compression and prompt-improvement tests complete.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
