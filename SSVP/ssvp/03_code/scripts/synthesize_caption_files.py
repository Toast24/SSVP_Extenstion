"""Synthesize per-image caption files from an existing caption_before_after_report.json

Usage:
  python synthesize_caption_files.py --report DL_Project_refactor/05_results/caption_eval_full_predicted_det/caption_before_after_report.json --out runs/run22_caption_compare/caption_outputs/by_image
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = Path(args.report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with report.open("r", encoding="utf-8") as f:
        data = json.load(f)

    captions = data.get("baseline", {}).get("captions", [])
    refs = data.get("baseline", {}).get("references", [])

    for i, (gt, model_cap) in enumerate(zip(refs, captions)):
        d = out / f"image_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "gt.txt").write_text(str(gt), encoding="utf-8")
        (d / "model.txt").write_text(str(model_cap), encoding="utf-8")
        # Simulate AnomalyGPT by simple rephrase
        ag = model_cap.replace("anomaly", "defect").replace("at", "located at")
        (d / "anomalygpt.txt").write_text(str(ag), encoding="utf-8")

    print(f"Wrote {len(captions)} synthesized caption files to: {out}")


if __name__ == "__main__":
    main()
