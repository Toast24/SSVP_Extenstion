"""
Compare candidate run metrics against a baseline run with a max-drop tolerance gate.

Usage:
    python 03_code/scripts/compare_against_baseline.py \
        --baseline_dir 05_results/ablations/run21_resplit_15es \
        --candidate_dir 05_results/ablations/run31_distill_53m_15ep \
    --max_drop 7.5 \
        --report_path 05_results/ablations/run31_distill_53m_15ep/comparison_to_run21.json
"""

import argparse
import json
import os
from typing import Dict, Tuple


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_clean_metrics(results_json: Dict) -> Dict[str, float]:
    overall = results_json.get("overall", {})
    image = overall.get("image_level", {})
    pixel = overall.get("pixel_level", {})

    return {
        "clean.image_auroc": float(image.get("auroc", 0.0)),
        "clean.image_f1_max": float(image.get("f1_max", 0.0)),
        "clean.image_ap": float(image.get("ap", 0.0)),
        "clean.pixel_auroc": float(pixel.get("auroc", 0.0)),
        "clean.pixel_pro": float(pixel.get("pro", 0.0)),
        "clean.pixel_ap": float(pixel.get("ap", 0.0)),
    }


def _extract_noise_metrics(robustness_json: Dict) -> Dict[str, float]:
    noisy = robustness_json.get("noisy", {})
    image = noisy.get("image_level", {})
    pixel = noisy.get("pixel_level", {})

    return {
        "noisy.image_auroc": float(image.get("auroc", 0.0)),
        "noisy.image_f1_max": float(image.get("f1_max", 0.0)),
        "noisy.image_ap": float(image.get("ap", 0.0)),
        "noisy.pixel_auroc": float(pixel.get("auroc", 0.0)),
        "noisy.pixel_pro": float(pixel.get("pro", 0.0)),
        "noisy.pixel_ap": float(pixel.get("ap", 0.0)),
    }


def _build_metric_set(run_dir: str) -> Dict[str, float]:
    results_path = os.path.join(run_dir, "eval_results", "results.json")
    noise_path = os.path.join(run_dir, "noise_eval", "robustness_report.json")

    results = _load_json(results_path)
    metrics = _extract_clean_metrics(results)

    if os.path.exists(noise_path):
        robustness = _load_json(noise_path)
        metrics.update(_extract_noise_metrics(robustness))

    return metrics


def _compare_metrics(
    baseline: Dict[str, float], candidate: Dict[str, float], max_drop: float
) -> Tuple[bool, Dict[str, Dict[str, float]], str, float]:
    shared_keys = sorted(set(baseline.keys()) & set(candidate.keys()))
    if not shared_keys:
        raise RuntimeError("No shared metrics found between baseline and candidate.")

    comparisons: Dict[str, Dict[str, float]] = {}
    gate_ok = True
    worst_key = ""
    worst_delta = float("inf")

    for key in shared_keys:
        b = baseline[key]
        c = candidate[key]
        delta = c - b
        comparisons[key] = {
            "baseline": b,
            "candidate": c,
            "delta": delta,
        }

        if delta < worst_delta:
            worst_delta = delta
            worst_key = key

        if delta < -max_drop:
            gate_ok = False

    return gate_ok, comparisons, worst_key, worst_delta


def main():
    parser = argparse.ArgumentParser(description="Compare run metrics to baseline with a max-drop gate")
    parser.add_argument("--baseline_dir", required=True, help="Baseline run directory")
    parser.add_argument("--candidate_dir", required=True, help="Candidate run directory")
    parser.add_argument("--max_drop", type=float, default=7.5, help="Maximum allowed drop per metric")
    parser.add_argument("--report_path", type=str, default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    baseline_metrics = _build_metric_set(args.baseline_dir)
    candidate_metrics = _build_metric_set(args.candidate_dir)

    gate_ok, comparisons, worst_key, worst_delta = _compare_metrics(
        baseline_metrics,
        candidate_metrics,
        max_drop=args.max_drop,
    )

    print("\n=== Baseline Comparison ===")
    print(f"Baseline:  {args.baseline_dir}")
    print(f"Candidate: {args.candidate_dir}")
    print(f"Max allowed drop per metric: {-args.max_drop:.2f}")
    print(f"Worst delta: {worst_delta:.3f} ({worst_key})")
    print(f"Gate status: {'PASS' if gate_ok else 'FAIL'}")

    print("\nMetric deltas (candidate - baseline):")
    for key in sorted(comparisons.keys()):
        d = comparisons[key]
        print(f"  {key:20s}  base={d['baseline']:8.3f}  cand={d['candidate']:8.3f}  delta={d['delta']:8.3f}")

    if args.report_path:
        report = {
            "baseline_dir": args.baseline_dir,
            "candidate_dir": args.candidate_dir,
            "max_drop": float(args.max_drop),
            "gate_pass": bool(gate_ok),
            "worst_metric": worst_key,
            "worst_delta": float(worst_delta),
            "comparisons": comparisons,
        }
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {args.report_path}")

    if not gate_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
