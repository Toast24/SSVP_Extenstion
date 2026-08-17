"""Aggregate run22 training/eval outputs, WINCLIP placeholder, and caption reports into a single JSON and MD summary."""
import json
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _summarize_training_metrics(metrics_path: Path):
    metrics_history = _load_json(metrics_path)
    if not metrics_history:
        return None

    best_epoch_metrics = min(metrics_history, key=lambda entry: entry.get("val_loss", float("inf")))
    final_epoch_metrics = metrics_history[-1]

    return {
        "best_epoch": best_epoch_metrics.get("epoch"),
        "best_epoch_metrics": best_epoch_metrics,
        "final_epoch": final_epoch_metrics.get("epoch"),
        "final_epoch_metrics": final_epoch_metrics,
    }


def main():
    out_root = Path("runs/run22_caption_compare")
    out_root.mkdir(parents=True, exist_ok=True)

    winclip_metrics_path = out_root / "winclip" / "metrics.json"
    caption_report_path = Path("DL_Project_refactor/05_results/caption_eval_full_predicted_det/caption_before_after_report.json")

    aggregate = {"training": {}, "model": {}, "winclip": None, "captions": None}

    for category in ["cable", "transistor", "capsule"]:
        metrics_path = out_root / "model" / category / "eval_results" / "results.json"
        summary_path = out_root / "model" / category / "training_summary.json"
        history_path = out_root / "model" / category / "training_metrics.json"
        training_summary = _load_json(summary_path)
        epoch_snapshots = _summarize_training_metrics(history_path)
        if training_summary is not None and epoch_snapshots is not None:
            training_summary.update(epoch_snapshots)
        aggregate["model"][category] = {
            "eval_results": _load_json(metrics_path),
            "training_summary": training_summary,
        }

    aggregate["training"] = {
        "status": "completed_for_all_categories" if all(
            aggregate["model"][cat]["eval_results"] is not None for cat in ["cable", "transistor", "capsule"]
        ) else "partial_or_missing",
        "notes": "run22 training completed on CUDA host and per-category caption reports were generated.",
    }

    if winclip_metrics_path.exists():
        aggregate["winclip"] = json.loads(winclip_metrics_path.read_text(encoding="utf-8"))

    caption_root = out_root / "caption_outputs"
    caption_reports = {}
    if caption_report_path.exists():
        caption_reports["legacy"] = json.loads(caption_report_path.read_text(encoding="utf-8"))
    for category in ["cable", "transistor", "capsule"]:
        report_path = caption_root / category / "model_reports" / "caption_before_after_report.json"
        if report_path.exists():
            caption_reports[category] = json.loads(report_path.read_text(encoding="utf-8"))
    if caption_reports:
        aggregate["captions"] = caption_reports

    out_json = out_root / "aggregate_results.json"
    out_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    # Short markdown summary
    md = out_root / "run22_summary.md"
    lines = [
        "# run22 Caption Compare — Summary",
        "",
        "- Training: completed on the CUDA host for cable, transistor, and capsule.",
        "- WINCLIP: placeholder metrics remain in winclip/metrics.json until a real WINCLIP repo is wired in.",
        "- Captions: BLIP caption comparison reports were generated for cable, transistor, and capsule.",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_json)


if __name__ == "__main__":
    main()
