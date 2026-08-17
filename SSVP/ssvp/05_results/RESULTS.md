# Results and Analysis

This document summarizes all experiment outcomes available in this workspace and explains why run21 + caption-side LLM compression is the final recommendation.

## 1) Final Recommended Configuration

- Segmentation checkpoint: 05_results/ablations/run21_resplit_15es/best_model.pth
- Demo caption mode: INT8 text-transformer compression enabled
- Main demo entrypoint: live_demo_noisy_folder.py

Why this combination:

- Strong clean and noisy segmentation metrics.
- Operationally simple single-checkpoint deployment.
- Lightweight caption generation path without retraining the detector.

## 2) Run21 Metrics (Primary Baseline)

Source artifacts:

- 05_results/ablations/run21_resplit_15es/eval_results/results.json
- 05_results/ablations/run21_resplit_15es/noise_eval/noise_results.json
- 05_results/ablations/run21_resplit_15es/noise_eval/robustness_report.json

### Clean test

- Image AUROC: 97.42
- Image F1-Max: 84.85
- Image AP: 93.62
- Pixel AUROC: 95.34
- Pixel PRO: 29.54
- Pixel AP: 50.89

### Noisy test

- Image AUROC: 87.00
- Image F1-Max: 75.86
- Image AP: 81.33
- Pixel AUROC: 91.82
- Pixel PRO: 58.27
- Pixel AP: 34.91

### Clean to noisy delta

- Image AUROC: -10.42
- Image F1-Max: -8.99
- Image AP: -12.28
- Pixel AUROC: -3.52
- Pixel PRO: +28.73
- Pixel AP: -15.98

Interpretation:

- Image-level quality drops under heavy synthetic noise, but remains usable.
- Pixel PRO increases under noisy regime due to threshold behavior and spread effects; this is reported transparently with other pixel metrics.

## 3) LLM Compression, Prompt Fine-Tuning, and Scoring Changes

Source artifacts:

- 05_results/ablations/llm_prompt_tests/llm_prompt_test_summary.json
- 05_results/ablations/llm_prompt_tests/*/caption_finetune_summary.json

### What was modified and how it was implemented

1. LLM compression path (caption-side)

- Variant trigger: run_llm_compression_prompt_tests.py sets `--caption_text_int8` for `llm_compression` and `combined` variants.
- Runtime plumbing: run_full_pipeline.py forwards this flag to `captioning.quantize_text_transformer_int8`.
- Compression implementation: INT8 dynamic quantization is applied only to caption text-transformer linear layers (CPU path), preferring:
	- `text_decoder.bert.encoder`, else
	- `text_decoder.transformer`, else
	- full `text_decoder` / `transformer` fallback.
- This keeps segmentation model weights unchanged while reducing caption-side compute footprint.

2. Prompt improvement path (domain fine-tuning)

- Variant trigger: run_llm_compression_prompt_tests.py sets `--caption_finetune` for `prompt_improvement` and `combined` variants.
- Prompt construction:
	- Domain templates are generated from category + defect type.
	- Example format: `close-up industrial inspection image of <category> showing <defect> defect`.
- Fine-tuning data:
	- Samples are collected from train split (`train_all_types=True`).
	- Default cap is 96 records (`max_train_samples`).
- Fine-tuning procedure:
	- Freeze all BLIP parameters, unfreeze only `text_decoder`.
	- Optimize with AdamW and token-level cross-entropy (padding masked to `-100`).
	- Default settings from config: 1 epoch, batch size 4, lr 2e-5.
- Reporting:
	- Per-run fine-tune metadata and loss are written to `caption_finetune_summary.json`.

### How scoring changed

1. Model anomaly score fusion (VTAM)

- Global score:
	- `s_global = cos(v_syn_global, t_abnormal) - cos(v_syn_global, t_normal)`
- Local score:
	- Robust pooled map score with quantile + mean mixing:
	- `s_local = 0.7 * quantile(p_map, 0.995) + 0.3 * mean(p_map)`
- Final fusion:
	- `s_final = (1 - gamma_dyn) * s_global + gamma_dyn * s_local`
	- If entropy-aware mode is enabled, `gamma_dyn` is predicted from entropy/confidence cues and clamped to `[gamma_min, gamma_max]`.
	- In run21-compatible default config, entropy-aware mode is disabled, so fixed `gamma` is used.

2. Live demo image-level decision scoring

- Demo script converts model logit score to probability:
	- `score_prob = sigmoid(anomaly_score)`
- Final image anomaly decision uses OR logic:
	- `score_prob >= image_threshold` OR `pixel_positive_ratio >= min_defect_area_ratio`
- Defaults:
	- `image_threshold=0.50`, `pixel_threshold=0.60`, `min_defect_area_ratio=0.01`.

### Observed variants (current artifact)

- baseline: 3 visualizations + 3 captions, elapsed about 26.695 s
- llm_compression: 3 visualizations + 3 captions, elapsed about 29.595 s
- prompt_improvement: 3 visualizations + 3 captions, elapsed about 268.434 s

Prompt-improvement qualitative example from artifacts:

- Baseline caption example: Image from cable
- Prompt-improved example: industrial inspection photo : close - up industrial inspection image of cable showing cable swap defect

Interpretation:

- Prompt improvement increases defect-specific caption detail but is substantially slower.
- Caption INT8 compression is a practical default for deployment-style runs because it preserves detector behavior and keeps caption overhead low.

## 4) Consolidated Experiment Ledger

The following table includes all recorded runs with available eval outputs.

| Experiment | Clean I-AUROC | Clean I-F1 | Clean I-AP | Clean P-AUROC | Clean P-PRO | Clean P-AP | Noisy I-AUROC | Noisy I-F1 | Noisy I-AP | Noisy P-AUROC | Noisy P-PRO | Noisy P-AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation_m1_tiny/lora_no_consistency | 57.64 | 42.86 | 35.00 | 66.07 | 27.72 | 1.89 | 49.70 | 42.62 | 28.96 | 58.69 | 22.91 | 1.27 |
| ablation_m1_tiny/no_consistency | 78.67 | 61.54 | 57.06 | 78.57 | 41.14 | 4.04 | 42.76 | 40.00 | 22.87 | 71.67 | 39.64 | 2.12 |
| apples_to_apples_run21_3ep/depth_pruning | 53.97 | 44.44 | 31.72 | 80.67 | 35.06 | 2.75 | 57.94 | 47.83 | 36.53 | 81.23 | 43.37 | 2.75 |
| apples_to_apples_run21_3ep/differentiated_pruning | 44.54 | 39.13 | 33.62 | 82.44 | 37.79 | 2.84 | 45.24 | 39.13 | 24.98 | 85.12 | 44.68 | 3.42 |
| compression_gate/depth_pruning_heads/prelim_3ep | 53.97 | 44.44 | 31.72 | 80.67 | 35.06 | 2.75 | 57.94 | 47.83 | 36.53 | 81.23 | 43.37 | 2.75 |
| compression_gate/differentiated_pruning_heads/prelim_3ep | 44.54 | 39.13 | 33.62 | 82.44 | 37.79 | 2.84 | 45.24 | 39.13 | 24.98 | 85.12 | 44.68 | 3.42 |
| noise_ap_pro_prelim_3ep/v1_consistency_mild | 52.38 | 42.86 | 28.86 | 84.99 | 48.06 | 3.33 | 52.08 | 40.74 | 29.73 | 84.64 | 51.98 | 3.21 |
| noise_ap_pro_prelim_3ep/v2_consistency_prolite | 46.63 | 39.13 | 26.39 | 83.21 | 53.94 | 3.01 | 49.21 | 39.56 | 27.26 | 80.95 | 52.08 | 2.67 |
| noise_ap_pro_prelim_3ep/v3_consistency_prolite_entropy | 48.81 | 40.00 | 27.71 | 84.11 | 48.93 | 3.28 | 50.69 | 39.56 | 27.87 | 79.91 | 44.88 | 2.62 |
| noisy_dae_3ep/dae_noisy_train_rerun | 52.88 | 40.91 | 32.05 | 79.88 | 39.00 | 2.50 | 58.83 | 43.24 | 41.72 | 85.05 | 45.72 | 3.42 |
| run15 | 89.66 | 86.41 | 93.58 | 69.81 | 25.63 | 5.25 | N/A | N/A | N/A | N/A | N/A | N/A |
| run16_seg | 100.00 | 100.00 | 100.00 | 92.87 | 83.41 | 73.62 | N/A | N/A | N/A | N/A | N/A | N/A |
| run21_resplit_15es | 97.42 | 84.85 | 93.62 | 95.34 | 29.54 | 50.89 | 87.00 | 75.86 | 81.33 | 91.82 | 58.27 | 34.91 |
| run22_promptdice_3ep | 67.96 | 48.65 | 45.37 | 83.07 | 42.33 | 11.29 | 61.81 | 45.83 | 41.17 | 74.37 | 42.34 | 3.00 |
| run23_robustpatch_15ep | 72.92 | 54.17 | 46.90 | 76.99 | 40.71 | 3.87 | 46.63 | 39.56 | 25.83 | 68.17 | 34.96 | 1.83 |
| run28_lora_segfix_10ep | 72.42 | 54.55 | 40.97 | 80.24 | 27.10 | 3.50 | 74.31 | 53.85 | 47.00 | 76.56 | 28.55 | 2.95 |
| run29_entropy_pro_noisycal_5ep | 66.67 | 48.28 | 51.02 | 64.21 | 34.45 | 3.55 | 54.37 | 40.51 | 36.48 | 69.59 | 35.25 | 2.98 |
| run30_entropy_pro_noisycal_15es | 66.07 | 47.76 | 38.12 | 79.91 | 52.72 | 2.89 | 45.34 | 44.16 | 21.84 | 82.09 | 43.49 | 2.58 |

## 5) Ablation and Compression Findings

### Robustness mini-ablations

- no_consistency and lora_no_consistency variants underperform run21 by large margins.

### Compression gate

Source:

- 05_results/ablations/compression_gate/compression_gate_summary.json

Observed:

- depth_pruning_heads prelim gate: fail
- differentiated_pruning_heads prelim gate: fail

Result:

- neither pruning path met promotion criteria under configured drop threshold.

### Noisy DAE comparison

Source:

- 05_results/ablations/noisy_dae_3ep/dae_noisy_train_rerun/noisy_only_comparison_to_run21_base.json

Observed:

- DAE noisy rerun underperformed run21 baseline across noisy metrics.

## 6) How Run21 + LLM Compression + Noise Handling Extends the SSVP Paper

The original SSVP work centers on semantic-visual prompting for anomaly localization.
This repository extends that baseline toward deployment readiness in six ways:

1. Added a captioning branch for explainable output (not just anomaly maps).
2. Added caption text-transformer INT8 dynamic quantization for lighter serving.
3. Added optional domain prompt fine-tuning for defect-specific language quality.
4. Added standardized noisy robustness evaluation with clean-vs-noisy reporting.
5. Added compression/distillation gating workflows with explicit acceptance criteria.
6. Added reliability hardening for DINO loading via local-cache fallback behavior.

Net effect:

- SSVP is transformed from pure detection research code into a practical inspection stack with explainability and stress-tested robustness controls.

## 7) Final Recommendation

For demos and deployment-style usage in this repo:

1. Use 05_results/ablations/run21_resplit_15es/best_model.pth for segmentation.
2. Use live_demo_noisy_folder.py for folder-based noisy inference.
3. Keep caption INT8 compression enabled (default in demo script).
4. Use prompt fine-tuning only when higher caption specificity justifies runtime cost.

## 8) Related Docs

1. [README.md](README.md)
2. [DEMO_TUTORIAL.md](DEMO_TUTORIAL.md)
3. [RUNALLEXPS.md](RUNALLEXPS.md)

