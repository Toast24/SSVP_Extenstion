# Clean vs Noisy Evaluation Summary

Date: 2026-05-28

## Scope
This summary combines:
1. WinCLIP zero-shot evaluation on MVTec AD categories `cable`, `transistor`, `capsule` for clean and noisy conditions.
2. Existing SSVP run outputs for:
   - `run21_resplit_15es` (requested as run21_15ep alias)
   - `transistor_run21_15es`
   - `capsule_run21_15es`
   each with clean and noisy metrics.

## Training and Testing Approach

### SSVP (trained model family)
- Training: supervised training was run with repository training scripts (`03_code/scripts/train.py`) and saved checkpoints under `05_results/ablations/*`.
- Clean testing: `inference.py` generated clean metrics from each checkpoint.
- Noisy testing: `evaluate_noise_robustness.py` generated noisy metrics (`noise_eval/noise_results.json`).
- Reported metrics: image-level AUROC/F1-Max/AP and pixel-level AUROC/PRO/AP.

### WinCLIP (zero-shot, no fine-tuning)
- Implementation: official WinCLIP code under `external/WinCLIP`.
- Inference script: `tools/winclip_eval.py`.
- Core settings:
  - zero-shot mode (`k-shot = 0`)
  - multi-scale windows `[2, 3]`
  - seed `42`
  - preprocessing used in the run: resize `240`, center-crop `240` (required by this backbone checkpoint shape)
- Clean eval: direct test split inference.
- Noisy eval: synthetic Gaussian noise injected at inference time with:
  - `noise_std = 0.20`
  - `noise_p = 1.0` (all test images)

## WinCLIP Results

### Clean
| Category | I-AUROC | I-F1 | I-AP | P-AUROC | P-PRO | P-AP |
|---|---:|---:|---:|---:|---:|---:|
| cable | 44.64 | 76.35 | 56.97 | 49.43 | 29.41 | 3.02 |
| transistor | 71.71 | 67.39 | 56.00 | 71.00 | 0.00 | 12.47 |
| capsule | 54.33 | 90.46 | 85.42 | 84.39 | 0.00 | 5.76 |
| mean | 56.89 | 78.07 | 66.13 | 68.27 | 9.80 | 7.08 |

### Noisy
| Category | I-AUROC | I-F1 | I-AP | P-AUROC | P-PRO | P-AP |
|---|---:|---:|---:|---:|---:|---:|
| cable | 46.42 | 76.03 | 63.84 | 45.82 | 28.18 | 2.88 |
| transistor | 42.42 | 57.55 | 41.46 | 62.48 | 0.00 | 10.94 |
| capsule | 56.48 | 90.83 | 87.64 | 76.13 | 0.00 | 5.31 |
| mean | 48.44 | 74.80 | 64.31 | 61.48 | 9.39 | 6.38 |

## Combined Artifacts Written
- `ssvp/05_results/ablations/combined_results.json`
  - appended `run21_15ep` alias entry
  - appended `winclip_zero_shot` clean/noisy section
- `results/winclip/summary_winclip.json`
  - now stores both clean and noisy WinCLIP summaries
- This document:
  - `ssvp/05_results/ablations/CLEAN_NOISY_EVAL_SUMMARY.md`

## Notes
- The requested name `run21_15ep` was mapped to existing `run21_resplit_15es` outputs.
- WinCLIP noisy results are based on synthetic Gaussian corruption applied during inference, not a separate hand-labeled noisy dataset split.
