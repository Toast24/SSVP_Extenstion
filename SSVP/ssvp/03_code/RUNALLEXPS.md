# Run All Experiments (Individually)

This guide provides command-level instructions for running each experiment script in this repository.

## 1) One-Time Environment Setup

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r 03_code/requirements.txt
```

## 2) One-Time Data Preparation

```powershell
python 03_code/scripts/prepare_cable_split.py --source 04_data/datasets/cable --output_root 04_data/datasets/cable_resplit --seed 42
```

## 3) Baseline End-to-End Run (Run21 Recipe)

```powershell
python 03_code/scripts/run_full_pipeline.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_dir 05_results/ablations/run21_resplit_15es --epochs 15 --num_vis_samples 5
```

## 4) Train Only

```powershell
python 03_code/scripts/train.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/train_only_run
```

## 5) Clean Inference Only

```powershell
python 03_code/scripts/inference.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/run21_resplit_15es/eval_results --visualize
```

## 6) Noisy Robustness Evaluation Only

```powershell
python 03_code/scripts/evaluate_noise_robustness.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --data_root 04_data/datasets/cable_resplit --categories cable --output_dir 05_results/ablations/run21_resplit_15es/noise_eval --clean_results 05_results/ablations/run21_resplit_15es/eval_results/results.json --calibration_file 05_results/ablations/run21_resplit_15es/calibration_thresholds.json
```

## 7) Main Live Demo (Run21 + LLM Compression)

```powershell
python 03_code/scripts/live_demo_noisy_folder.py --input_folder 04_data/datasets/cable/test/combined --output_dir 05_results/ablations/live_demo --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --category cable --recursive
```

Optional prompt-finetune mode:

```powershell
python 03_code/scripts/live_demo_noisy_folder.py --input_folder 04_data/datasets/cable/test/combined --output_dir 05_results/ablations/live_demo_prompt --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --category cable --recursive --caption_finetune
```

## 8) LLM Compression + Prompt Variant Tests

Run all variants:

```powershell
python 03_code/scripts/run_llm_compression_prompt_tests.py --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/llm_prompt_tests --num_vis_samples 3 --variants baseline llm_compression prompt_improvement combined
```

Run only a subset (example):

```powershell
python 03_code/scripts/run_llm_compression_prompt_tests.py --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/llm_prompt_tests_fast --num_vis_samples 3 --variants baseline llm_compression
```

## 9) Ablation Matrix Experiments

Run all defined ablations:

```powershell
python 03_code/scripts/run_ablation_matrix.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/ablation_m1 --epochs 3 --num_vis_samples 5
```

Run a single ablation variant (example no_consistency):

```powershell
python 03_code/scripts/run_ablation_matrix.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/ablation_single --epochs 3 --variants no_consistency
```

Other supported ablation variant names:

- no_tta
- no_robust_postproc
- weak_aug
- lora_no_consistency

## 10) Head Sanity Tests (Pruning/LoRA)

```powershell
python 03_code/scripts/run_head_sanity_tests.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_root 05_results/ablations/head_sanity_3ep --epochs 3
```

## 11) Compression Gate Pipeline

Run all defined compression/distillation gate methods:

```powershell
python 03_code/scripts/run_compression_gate_pipeline.py --baseline_dir 05_results/ablations/run21_resplit_15es --data_root 04_data/datasets/cable_resplit --categories cable --output_root 05_results/ablations/compression_gate --prelim_epochs 3 --full_epochs 15 --max_drop 7.0
```

Run only prelim gate stage:

```powershell
python 03_code/scripts/run_compression_gate_pipeline.py --baseline_dir 05_results/ablations/run21_resplit_15es --data_root 04_data/datasets/cable_resplit --categories cable --output_root 05_results/ablations/compression_gate_prelim --prelim_epochs 3 --full_epochs 15 --max_drop 7.0 --prelim_only
```

Run one compression-gate variant only (example):

```powershell
python 03_code/scripts/run_compression_gate_pipeline.py --baseline_dir 05_results/ablations/run21_resplit_15es --data_root 04_data/datasets/cable_resplit --categories cable --output_root 05_results/ablations/compression_gate_depth --variants depth_pruning_heads
```

Other supported variant names:

- differentiated_pruning_heads
- student_53m_distill
- student_43m_redistill

## 12) Staged Distillation Pipeline

```powershell
python 03_code/scripts/run_staged_distillation.py
```

## 13) Baseline Comparison Gate Utility

```powershell
python 03_code/scripts/compare_against_baseline.py --baseline_dir 05_results/ablations/run21_resplit_15es --candidate_dir 05_results/ablations/run31_distill_53m_15ep --max_drop 7.5 --report_path 05_results/ablations/run31_distill_53m_15ep/comparison_to_run21.json
```

Exit code:

- 0 = pass
- 2 = fail

## 14) Tensor Shape Contract Test

```powershell
python 03_code/scripts/test_shapes.py
```

## 15) Expected Summary Files

- 05_results/ablations/run21_resplit_15es/eval_results/results.json
- 05_results/ablations/run21_resplit_15es/noise_eval/robustness_report.json
- 05_results/ablations/llm_prompt_tests/llm_prompt_test_summary.json
- 05_results/ablations/ablation_m1/ablation_summary.json
- 05_results/ablations/head_sanity_3ep/sanity_summary.json
- 05_results/ablations/compression_gate/compression_gate_summary.json
- 05_results/ablations/staged_distillation_summary.json

## 16) Keep Only Final Checkpoint (Optional Cleanup)

```powershell
$best = (Resolve-Path "05_results/ablations/run21_resplit_15es/best_model.pth").Path
Get-ChildItem 05_results/ablations -Recurse -File -Filter *.pth |
  Where-Object { $_.FullName -ne $best } |
  Remove-Item -Force
```

Verification:

```powershell
Get-ChildItem 05_results/ablations -Recurse -File -Include *.pth,*.pt,*.ckpt | Select-Object -ExpandProperty FullName | Sort-Object
```

