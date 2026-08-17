# Scripts Index

Runnable entry points for `SSVP/ssvp/`. Run all commands from the **repo root** (`ssvp/`), not from this folder.

For full flags and examples see [`../README.md`](../README.md) and [`../RUNALLEXPS.md`](../RUNALLEXPS.md).

## Data prep

| Script | One-line purpose |
|--------|------------------|
| [`prepare_cable_split.py`](prepare_cable_split.py) | 70/20/10 train/test/val resplit for MVTec categories |

## Train and evaluate

| Script | One-line purpose |
|--------|------------------|
| [`train.py`](train.py) | Core SSVP training with early stopping and checkpoints |
| [`inference.py`](inference.py) | Clean-test metrics and optional visualizations |
| [`evaluate_noise_robustness.py`](evaluate_noise_robustness.py) | Heavy synthetic noise stress test |
| [`compare_against_baseline.py`](compare_against_baseline.py) | Pass/fail gate vs a baseline run directory |
| [`test_shapes.py`](test_shapes.py) | Architecture sanity check (run after model edits) |

## End-to-end workflows

| Script | One-line purpose |
|--------|------------------|
| [`run_full_pipeline.py`](run_full_pipeline.py) | Train → clean eval → noisy eval → caption exports |
| [`live_demo_noisy_folder.py`](live_demo_noisy_folder.py) | Demo on arbitrary image folders (recommended deployment path) |
| [`run_ablation_matrix.py`](run_ablation_matrix.py) | Compact robustness ablation grid |

## Compression and distillation

| Script | One-line purpose |
|--------|------------------|
| [`run_compression_gate_pipeline.py`](run_compression_gate_pipeline.py) | Gated two-stage compression |
| [`run_staged_distillation.py`](run_staged_distillation.py) | Student distillation with baseline gates |
| [`run_head_sanity_tests.py`](run_head_sanity_tests.py) | Short head-pruning sanity runs |
| [`run_llm_compression_prompt_tests.py`](run_llm_compression_prompt_tests.py) | Caption INT8 / prompt variant comparison |

## Baselines (outside this folder)

| Script | Location |
|--------|----------|
| WinCLIP evaluation | [`../../tools/winclip_eval.py`](../../tools/winclip_eval.py) |
| AnomalyCLIP | [`../external/AnomalyCLIP/`](../external/AnomalyCLIP/) |
| AnomalyGPT | [`../../external/AnomalyGPT/`](../../external/AnomalyGPT/) |

## Utilities

| Module | Purpose |
|--------|---------|
| [`path_utils.py`](path_utils.py) | `REPO_ROOT`, dataset/results path constants; call `ensure_import_paths()` at script top |

## Planned imports from REFAC

Caption evaluation and run22 scripts live in `SSVPREFAC/DL_Project_refactor/03_code/scripts/` until merged. See [`../../../../docs/MERGE_PLAN.md`](../../../../docs/MERGE_PLAN.md).

## Typical flow

```
prepare_cable_split → run_full_pipeline → live_demo_noisy_folder
                              ↓
                    compare_against_baseline (optional gate)
```

Recommended checkpoint: `05_results/ablations/run21_resplit_15es/best_model.pth`
