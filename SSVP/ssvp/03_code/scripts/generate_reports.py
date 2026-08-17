"""Aggregate outputs from training, WINCLIP, and caption comparison into markdown reports.

Produces:
  - runs/run22_caption_compare/run22_training_summary.md
  - runs/run22_caption_compare/winclip_vs_model.md
  - runs/run22_caption_compare/caption_comparison.md
  - runs/run22_caption_compare/changes_from_run21.md
"""
import argparse
import json
import os
from pathlib import Path


def write_training_summary(root, out_path):
    model_root = Path(root) / "model"
    lines = []
    lines.append("# run22 Training Summary")
    for cat_dir in sorted((Path(root) / "model").iterdir() if model_root.exists() else []):
        if not cat_dir.is_dir():
            continue
        lines.append(f"\n## Category: {cat_dir.name}")
        metrics = cat_dir / "metrics.csv"
        if metrics.exists():
            lines.append("\n**Metrics (csv):**")
            lines.append(f"- {metrics}")
        ckpt = cat_dir / "best.pt"
        lines.append(f"- Best checkpoint: {ckpt if ckpt.exists() else 'MISSING'}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_winclip_comparison(winclip_metrics_path, model_root, out_path):
    lines = ["# WINCLIP vs Model Comparison\n"]
    if os.path.exists(winclip_metrics_path):
        with open(winclip_metrics_path, "r", encoding="utf-8") as f:
            w = json.load(f)
        lines.append("## WINCLIP metrics summary")
        lines.append("```\n" + json.dumps(w, indent=2) + "\n```")
    else:
        lines.append("WINCLIP metrics not found.")

    # List model metrics
    lines.append("\n## Model metrics (per category)")
    for cat_dir in sorted(Path(model_root).iterdir() if Path(model_root).exists() else []):
        m = Path(cat_dir) / "metrics.csv"
        lines.append(f"- {cat_dir.name}: {m if m.exists() else 'MISSING'}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_caption_comparison(caption_outputs_dir, out_path):
    lines = ["# Caption Comparison\n"]
    by_img = Path(caption_outputs_dir) / "by_image"
    if not by_img.exists():
        lines.append("No per-image caption outputs found.")
    else:
        examples = sorted([p for p in by_img.iterdir() if p.is_dir()])[:10]
        for ex in examples:
            lines.append(f"\n## {ex.name}")
            for fn in ["gt.txt", "model.txt", "anomalygpt.txt"]:
                p = ex / fn
                if p.exists():
                    txt = p.read_text(encoding="utf-8").strip()
                    lines.append(f"- **{fn}**: {txt}")
                else:
                    lines.append(f"- **{fn}**: MISSING")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="runs/run22_caption_compare")
    args = parser.parse_args()

    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)

    write_training_summary(str(root), str(root / "run22_training_summary.md"))
    write_winclip_comparison(str(root / "winclip" / "metrics.json"), str(root / "model"), str(root / "winclip_vs_model.md"))
    write_caption_comparison(str(root / "caption_outputs"), str(root / "caption_comparison.md"))

    # Keep changes file as a minimal template if missing
    changelog = root / "changes_from_run21.md"
    if not changelog.exists():
        changelog.write_text("# Changes from run21 to run22_caption_compare\n\n- Experimental changes:\n  - deterministic splits for transistor/capsule\n  - epochs=15, early stopping patience=4\n\n- No architectural changes to encoder/decoder/caption branch (baseline preserved).\n")

    print("Reports generated under:", root)


if __name__ == "__main__":
    main()
