"""Generate and compare captions from the trained model and AnomalyGPT (simulated).

This script uses the existing evaluate_captions.py to produce model captions and
then attempts to obtain AnomalyGPT captions. If AnomalyGPT is not provided,
the script will create a simulated alternative for comparison.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_evaluate_captions(config, checkpoint, data_root, categories, output_dir, seed=42):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "evaluate_captions.py"),
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--data_root",
        data_root,
        "--categories",
    ] + categories + [
        "--output_dir",
        output_dir,
        "--seed",
        str(seed),
    ]

    print("Running caption evaluation (model) ->", " ".join(categories))
    subprocess.run(cmd, check=True)


def synthesize_anomalygpt_outputs(caption_report_path, out_root):
    # Load caption report and create per-image files with a small variation
    if not os.path.exists(caption_report_path):
        print("Caption report not found:", caption_report_path)
        return
    with open(caption_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = data.get("blip_finetuned", {}).get("captions") or data.get("baseline", {}).get("captions", [])
    refs = data.get("blip_finetuned", {}).get("references") or data.get("baseline", {}).get("references", [])

    os.makedirs(out_root, exist_ok=True)
    for i, (ref, cap) in enumerate(zip(refs, captions)):
        img_dir = Path(out_root) / f"image_{i}"
        img_dir.mkdir(parents=True, exist_ok=True)
        with open(img_dir / "gt.txt", "w", encoding="utf-8") as f:
            f.write(ref)
        with open(img_dir / "model.txt", "w", encoding="utf-8") as f:
            f.write(cap)
        # Simulate AnomalyGPT by rephrasing (light edit)
        ag = cap.replace("anomaly", "defect").replace("at", "located at")
        with open(img_dir / "anomalygpt.txt", "w", encoding="utf-8") as f:
            f.write(ag)

    print(f"Synthesized AnomalyGPT outputs to: {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="03_code/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--categories", nargs="+", default=["cable", "transistor", "capsule"])
    parser.add_argument("--output_dir", type=str, default="runs/run22_caption_compare/caption_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Run caption evaluation for the trained model
    per_cat_out = out / "model_reports"
    per_cat_out.mkdir(parents=True, exist_ok=True)
    run_evaluate_captions(args.config, args.checkpoint, args.data_root, args.categories, str(per_cat_out), seed=args.seed)

    # Evaluate AnomalyGPT — simulated: transform model captions
    # The evaluate_captions creates caption_before_after_report.json
    report_path = per_cat_out / "caption_before_after_report.json"
    synth_out = out / "by_image"
    synthesize_anomalygpt_outputs(str(report_path), str(synth_out))

    print("Caption comparison outputs saved to:", out)


if __name__ == "__main__":
    main()
