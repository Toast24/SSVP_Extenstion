# SSVP: Robust Industrial Anomaly Segmentation + Captioning

This repository implements an extended SSVP workflow for industrial visual inspection on MVTec AD cable data, with three practical goals:

1. High-quality anomaly segmentation.
2. Robustness under realistic image noise.
3. Human-readable defect captions for demo and ops workflows.

The final recommended deployment path in this repo is:

- Segmentation model: run21 checkpoint.
- Caption path: LLM text-side INT8 compression.
- Demo path: noisy-folder live demo that outputs overlays, masks, heatmaps, and captions.

## Why This Project Matters

Industrial anomaly systems are most useful when they are both accurate and explainable.

- Accuracy: segment defect regions reliably.
- Robustness: keep useful performance under noisy conditions.
- Explainability: produce concise textual descriptions alongside visual heatmaps.

This project is suitable for:

- Factory visual QA pilots.
- Defect triage dashboards.
- Dataset and model ablation studies.
- Compression and distillation experimentation for constrained deployments.

## Dataset (Official MVTec Link)

Use the official MVTec AD download page:

- https://www.mvtec.com/company/research/datasets/mvtec-ad

After download, place the cable category under 04_data/datasets/cable so this repo can read paths like:

- 04_data/datasets/cable/train/good/
- 04_data/datasets/cable/test/good/
- 04_data/datasets/cable/test/<defect_type>/
- 04_data/datasets/cable/ground_truth/<defect_type>/

## Documentation Map

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project premise, methodology, script map |
| [scripts/README.md](scripts/README.md) | Scripts grouped by purpose (quick index) |
| [demo_instructions.md](../06_demo/demo_instructions.md) | From-scratch demo (run21 + LLM compression) |
| [RUNALLEXPS.md](RUNALLEXPS.md) | Command cookbook for each experiment |
| [RESULTS.md](../05_results/RESULTS.md) | Consolidated outcomes and run21 recommendation |
| [../../docs/STRUCTURE.md](../../docs/STRUCTURE.md) | Monorepo layout (SSVP vs SSVPREFAC) |
| [../../docs/MERGE_PLAN.md](../../docs/MERGE_PLAN.md) | Consolidation roadmap |

## Methodology Overview

Core modeling follows SSVP components (HSVS, VCPG, VTAM) and extends them operationally with:

- Train/val anti-leakage protocol and validation-driven thresholds.
- Explicit noisy robustness evaluation.
- Captioning branch with optional domain fine-tuning.
- Caption text-transformer INT8 dynamic quantization for lighter inference.
- Structured ablation, compression gating, and staged distillation scripts.

## Runnable Files: Significance, Use Case, and How To Run

All runnable entry scripts are under 03_code/scripts.

### 1) prepare_cable_split.py

- Significance: creates a clean 70/20/10 train/test/val cable split with aligned masks.
- Use case: first step before reproducible training and evaluation.
- Run:

```powershell
python 03_code/scripts/prepare_cable_split.py --source 04_data/datasets/cable --output_root 04_data/datasets/cable_resplit --seed 42
```

### 2) train.py

- Significance: core SSVP training with validation monitoring, early stopping, checkpointing, and calibration export.
- Use case: train a model only (without full pipeline orchestration).
- Run:

```powershell
python 03_code/scripts/train.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/my_train_run
```

### 3) inference.py

- Significance: evaluates clean test performance and writes full metrics JSON.
- Use case: evaluate a trained checkpoint on clean test split.
- Run:

```powershell
python 03_code/scripts/inference.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/my_train_run/best_model.pth --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/my_train_run/eval_results --visualize
```

### 4) evaluate_noise_robustness.py

- Significance: evaluates robustness under synthetic heavy noise and reports clean-to-noisy deltas.
- Use case: stress-test deployment reliability.
- Run:

```powershell
python 03_code/scripts/evaluate_noise_robustness.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/my_train_run/best_model.pth --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/my_train_run/noise_eval --clean_results 05_results/ablations/my_train_run/eval_results/results.json
```

### 5) run_full_pipeline.py

- Significance: orchestrates train + clean eval + noisy eval + captioned visualization exports.
- Use case: default end-to-end run path.
- Run:

```powershell
python 03_code/scripts/run_full_pipeline.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_dir 05_results/ablations/run21_resplit_15es --epochs 15 --num_vis_samples 5
```

### 6) live_demo_noisy_folder.py

- Significance: main deployment-style demo on arbitrary noisy folder inputs.
- Use case: generate segmentation artifacts and caption text for new images.
- Run (INT8 caption compression enabled by default):

```powershell
python 03_code/scripts/live_demo_noisy_folder.py --input_folder 04_data/datasets/cable/test/combined --output_dir 05_results/ablations/live_demo --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --category cable --recursive
```

### 7) run_ablation_matrix.py

- Significance: executes compact robustness ablations and aggregates clean/noisy metrics.
- Use case: investigate consistency, TTA, postproc, augmentation, and LoRA toggles.
- Run:

```powershell
python 03_code/scripts/run_ablation_matrix.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/ablation_m1 --epochs 3 --num_vis_samples 5
```

### 8) compare_against_baseline.py

- Significance: metric gate utility for candidate-vs-baseline decision.
- Use case: CI-like pass/fail quality check for any run output.
- Run:

```powershell
python 03_code/scripts/compare_against_baseline.py --baseline_dir 05_results/ablations/run21_resplit_15es --candidate_dir 05_results/ablations/run31_distill_53m_15ep --max_drop 7.5 --report_path 05_results/ablations/run31_distill_53m_15ep/comparison_to_run21.json
```

### 9) test_shapes.py

- Significance: validates module tensor-shape contracts for HSVS/VCPG/VTAM/losses.
- Use case: architecture sanity check after model edits.
- Run:

```powershell
python 03_code/scripts/test_shapes.py
```

### 10) evaluate_captions.py / caption_compare.py (ported from SSVPREFAC, see docs/MERGE_PLAN.md)

- Significance: BERTScore-based caption quality benchmark (baseline vs BLIP fine-tuned vs predicted-map synthesis); `caption_compare.py` wraps it to also emit simulated AnomalyGPT comparisons.
- Use case: quantify caption quality deltas and produce per-image comparison files.
- Run:

```powershell
python 03_code/scripts/evaluate_captions.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/caption_eval
```

### 11) train_run22.py + configs/run22_override.yaml (ported from SSVPREFAC)

- Significance: orchestrates the run22 caption-compare training flow (train_all_types, early stopping disabled, domain fine-tune) on top of the existing `run_full_pipeline.py`.
- Use case: reproduce the run22 experiment without hand-editing `default.yaml`.

### 12) run_anomalyclip_baseline.py (ported from SSVPREFAC)

- Significance: zero-shot AnomalyCLIP baseline runner wired to `03_code/external/AnomalyCLIP`.
- Use case: compare against an external zero-shot baseline (currently emits simulated results pending full AnomalyCLIP wiring).

### 13) generate_reports.py / aggregate_run22_results.py / synthesize_caption_files.py (ported from SSVPREFAC)

- Significance: aggregate and report on JSON results under `05_results/`.
- Use case: roll up multiple run outputs into summary reports.

### Deprecated scripts (`_deprecated/`)

Moved out of the active script list per `docs/MERGE_PLAN.md` Phase 2 (still runnable, just not part of the primary workflow):

- `run_compression_gate_pipeline.py`
- `run_head_sanity_tests.py`
- `run_llm_compression_prompt_tests.py`
- `run_staged_distillation.py`

## Recommended First-Time Flow

If you are new to this repo:

1. Follow [demo_instructions.md](../06_demo/demo_instructions.md) end to end.
2. Use [RUNALLEXPS.md](RUNALLEXPS.md) to run additional experiments.
3. Read [RESULTS.md](../05_results/RESULTS.md) for conclusions and final model recommendation.

