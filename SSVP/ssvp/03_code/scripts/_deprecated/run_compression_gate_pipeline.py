"""
Run supported parameter-compression methods with a two-stage gate:
1) prelim run at 3 epochs
2) if prelim passes max-drop gate vs baseline, run full 15 epochs

Gate rule:
  - Per-metric maximum allowed drop (candidate - baseline) >= -max_drop
  - Implemented via compare_against_baseline.py

Notes:
  - Methods are executed strictly one-by-one.
  - Uses a train-stage workaround: if training artifacts are already written but
    train.py remains alive (known calibration stall), the process is terminated
    and evaluation proceeds from best_model.pth.
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from path_utils import CONFIG_DIR, DATASETS_DIR, LOGS_DIR, SCRIPTS_DIR


TRAIN_SCRIPT = str(SCRIPTS_DIR / "train.py")
INFERENCE_SCRIPT = str(SCRIPTS_DIR / "inference.py")
NOISE_EVAL_SCRIPT = str(SCRIPTS_DIR / "evaluate_noise_robustness.py")
COMPARE_SCRIPT = str(SCRIPTS_DIR / "compare_against_baseline.py")

DEFAULT_RUN21_DIR = LOGS_DIR / "run21_resplit_15es"
DEFAULT_RUN31_DIR = LOGS_DIR / "run31_distill_53m_15ep"


UNSUPPORTED_METHODS = {
    "low_rank_factorization": "No native matrix-factorization module replacement is implemented.",
    "quantization": "No integrated train+eval quantized pipeline is implemented in current scripts.",
}


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def run_cmd(cmd):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def append_categories(cmd, categories):
    if categories:
        cmd += ["--categories", *categories]


def run_train_with_stall_workaround(
    py_exe,
    config_path,
    output_dir,
    data_root,
    categories,
    terminate_grace_sec=90,
    poll_sec=5,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "training_summary.json"
    best_ckpt_path = output_dir / "best_model.pth"

    cmd = [
        py_exe,
        TRAIN_SCRIPT,
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
        "--data_root",
        data_root,
    ]
    append_categories(cmd, categories)

    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)

    summary_seen_at = None
    terminated_after_summary = False

    while True:
        rc = proc.poll()
        if rc is not None:
            break

        if summary_path.exists() and best_ckpt_path.exists():
            if summary_seen_at is None:
                summary_seen_at = time.time()
            elif (time.time() - summary_seen_at) >= terminate_grace_sec:
                print(
                    "Detected training artifacts while train.py still running; "
                    "terminating process to avoid calibration stall.",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
                terminated_after_summary = True
                break

        time.sleep(poll_sec)

    final_rc = proc.poll()
    if final_rc is None:
        final_rc = 0

    # Non-zero return is tolerated if artifacts exist (manual termination path).
    if final_rc != 0 and not (summary_path.exists() and best_ckpt_path.exists()):
        raise subprocess.CalledProcessError(final_rc, cmd)

    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_ckpt_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")

    return {
        "best_checkpoint": str(best_ckpt_path),
        "training_summary": str(summary_path),
        "terminated_after_summary": bool(terminated_after_summary),
        "train_return_code": int(final_rc),
    }


def run_eval_pipeline(py_exe, config_path, checkpoint_path, run_dir, data_root, categories):
    eval_dir = run_dir / "eval_results"
    noise_dir = run_dir / "noise_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    infer_cmd = [
        py_exe,
        INFERENCE_SCRIPT,
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output_dir",
        str(eval_dir),
        "--data_root",
        data_root,
    ]
    append_categories(infer_cmd, categories)
    run_cmd(infer_cmd)

    noise_cmd = [
        py_exe,
        NOISE_EVAL_SCRIPT,
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output_dir",
        str(noise_dir),
        "--clean_results",
        str(eval_dir / "results.json"),
        "--data_root",
        data_root,
    ]
    append_categories(noise_cmd, categories)
    run_cmd(noise_cmd)


def compare_to_baseline(py_exe, baseline_dir, candidate_dir, max_drop, report_path):
    cmd = [
        py_exe,
        COMPARE_SCRIPT,
        "--baseline_dir",
        str(baseline_dir),
        "--candidate_dir",
        str(candidate_dir),
        "--max_drop",
        str(max_drop),
        "--report_path",
        str(report_path),
    ]

    try:
        run_cmd(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def collect_stage_metrics(run_dir):
    clean_path = run_dir / "eval_results" / "results.json"
    noisy_path = run_dir / "noise_eval" / "robustness_report.json"

    metrics = {}
    if clean_path.exists():
        clean = load_json(clean_path).get("overall", {})
        metrics["clean"] = {
            "image_ap": clean.get("image_level", {}).get("ap"),
            "image_auroc": clean.get("image_level", {}).get("auroc"),
            "pixel_pro": clean.get("pixel_level", {}).get("pro"),
            "pixel_ap": clean.get("pixel_level", {}).get("ap"),
        }

    if noisy_path.exists():
        noisy = load_json(noisy_path).get("noisy", {})
        metrics["noisy"] = {
            "image_ap": noisy.get("image_level", {}).get("ap"),
            "image_auroc": noisy.get("image_level", {}).get("auroc"),
            "pixel_pro": noisy.get("pixel_level", {}).get("pro"),
            "pixel_ap": noisy.get("pixel_level", {}).get("ap"),
        }

    return metrics


def make_methods():
    return [
        {
            "name": "depth_pruning_heads",
            "type": "structured_pruning",
            "base_config": str(DEFAULT_RUN21_DIR / "_temp_config.yaml"),
            "overrides": {
                "head_pruning": {
                    "enabled": True,
                    "mode": "depth",
                    "depth_keep_ratio": 0.75,
                },
                "lora": {"enabled": False},
            },
        },
        {
            "name": "differentiated_pruning_heads",
            "type": "structured_pruning",
            "base_config": str(DEFAULT_RUN21_DIR / "_temp_config.yaml"),
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
                "lora": {"enabled": False},
            },
        },
        {
            "name": "student_53m_distill",
            "type": "distillation",
            "base_config": str(CONFIG_DIR / "ssvp_student_53m_distill.yaml"),
            "overrides": {},
            "required_paths": [str(DEFAULT_RUN21_DIR / "best_model.pth")],
        },
        {
            "name": "student_43m_redistill",
            "type": "distillation",
            "base_config": str(CONFIG_DIR / "ssvp_student_43m_redistill.yaml"),
            "overrides": {},
            "required_paths": [str(DEFAULT_RUN31_DIR / "best_model.pth")],
        },
    ]


def prepare_stage_config(method, stage_epochs, out_cfg_path):
    cfg = load_yaml(method["base_config"])
    deep_update(cfg, method.get("overrides", {}))
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = int(stage_epochs)
    save_yaml(out_cfg_path, cfg)
    return out_cfg_path


def run_stage(
    py_exe,
    method,
    stage_name,
    stage_epochs,
    output_root,
    baseline_dir,
    data_root,
    categories,
    max_drop,
):
    method_root = output_root / method["name"]
    stage_dir = method_root / stage_name
    cfg_dir = output_root / "configs"
    cfg_path = cfg_dir / f"{method['name']}_{stage_name}.yaml"
    report_path = stage_dir / f"comparison_to_{Path(baseline_dir).name}.json"

    prepare_stage_config(method, stage_epochs, cfg_path)

    train_info = run_train_with_stall_workaround(
        py_exe=py_exe,
        config_path=cfg_path,
        output_dir=stage_dir,
        data_root=data_root,
        categories=categories,
    )

    run_eval_pipeline(
        py_exe=py_exe,
        config_path=cfg_path,
        checkpoint_path=Path(train_info["best_checkpoint"]),
        run_dir=stage_dir,
        data_root=data_root,
        categories=categories,
    )

    gate_pass = compare_to_baseline(
        py_exe=py_exe,
        baseline_dir=baseline_dir,
        candidate_dir=stage_dir,
        max_drop=max_drop,
        report_path=report_path,
    )

    return {
        "stage_name": stage_name,
        "epochs": int(stage_epochs),
        "config_path": str(cfg_path),
        "run_dir": str(stage_dir),
        "compare_report": str(report_path),
        "gate_pass": bool(gate_pass),
        "train_info": train_info,
        "metrics": collect_stage_metrics(stage_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run compression methods with prelim gate and full promotion")
    parser.add_argument("--baseline_dir", type=str, default=str(DEFAULT_RUN21_DIR))
    parser.add_argument("--data_root", type=str, default=str(DATASETS_DIR / "cable_resplit"))
    parser.add_argument("--categories", nargs="+", default=["cable"])
    parser.add_argument("--output_root", type=str, default=str(LOGS_DIR / "compression_gate"))
    parser.add_argument("--prelim_epochs", type=int, default=3)
    parser.add_argument("--full_epochs", type=int, default=15)
    parser.add_argument("--max_drop", type=float, default=7.0)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Optional subset of method names to run")
    parser.add_argument("--prelim_only", action="store_true",
                        help="Only run prelim stage and do not promote to full runs")
    args = parser.parse_args()

    py_exe = sys.executable
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    methods = make_methods()
    if args.variants:
        requested = set(args.variants)
        methods = [m for m in methods if m["name"] in requested]
        if not methods:
            raise ValueError("No matching variants found for --variants")

    summary = {
        "baseline_dir": args.baseline_dir,
        "data_root": args.data_root,
        "categories": args.categories,
        "max_drop": float(args.max_drop),
        "prelim_epochs": int(args.prelim_epochs),
        "full_epochs": int(args.full_epochs),
        "prelim_only": bool(args.prelim_only),
        "unsupported_methods": UNSUPPORTED_METHODS,
        "methods": {},
    }

    for method in methods:
        method_name = method["name"]
        print(f"\n=== Method: {method_name} ===", flush=True)
        method_entry = {
            "type": method.get("type"),
            "required_paths": method.get("required_paths", []),
            "prelim": None,
            "full": None,
            "promoted_to_full": False,
            "skipped": False,
            "skip_reason": None,
        }

        missing = [p for p in method.get("required_paths", []) if not Path(p).exists()]
        if missing:
            method_entry["skipped"] = True
            method_entry["skip_reason"] = f"Missing required paths: {missing}"
            summary["methods"][method_name] = method_entry
            print(method_entry["skip_reason"], flush=True)
            continue

        prelim_stage = run_stage(
            py_exe=py_exe,
            method=method,
            stage_name=f"prelim_{args.prelim_epochs}ep",
            stage_epochs=args.prelim_epochs,
            output_root=output_root,
            baseline_dir=args.baseline_dir,
            data_root=args.data_root,
            categories=args.categories,
            max_drop=args.max_drop,
        )
        method_entry["prelim"] = prelim_stage

        if prelim_stage["gate_pass"] and not args.prelim_only:
            full_stage = run_stage(
                py_exe=py_exe,
                method=method,
                stage_name=f"full_{args.full_epochs}ep",
                stage_epochs=args.full_epochs,
                output_root=output_root,
                baseline_dir=args.baseline_dir,
                data_root=args.data_root,
                categories=args.categories,
                max_drop=args.max_drop,
            )
            method_entry["full"] = full_stage
            method_entry["promoted_to_full"] = True

        summary["methods"][method_name] = method_entry

        summary_path = output_root / "compression_gate_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    summary_path = output_root / "compression_gate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCompression gate run complete.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
