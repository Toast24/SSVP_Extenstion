# Script Inventory

All runnable entry points in both trees. Paths are relative to each project's repo root (`SSVP/ssvp/` or `SSVPREFAC/DL_Project_refactor/`).

Legend: ✅ both trees · 🅰 SSVP only · 🅱 REFAC only · 🗄 REFAC deprecated · ⚠ diverged implementation

## Data preparation

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `03_code/scripts/prepare_cable_split.py` | 🅰 | — | 70/20/10 resplit for one or more MVTec categories (supports nested dirs) |
| `03_code/scripts/prepare_category_split.py` | — | 🅱 | Single-category resplit; **superseded by prepare_cable_split** |
| `03_code/scripts/split_dataset.py` | — | 🅱 | Generic split utility — review before porting |

## Training and evaluation

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `03_code/scripts/train.py` | ⚠ | ⚠ | Core SSVP training loop |
| `03_code/scripts/train_run22.py` | — | 🅱 | Run22 prompt-dice training variant |
| `03_code/scripts/inference.py` | ⚠ | ⚠ | Clean-test evaluation + metrics JSON |
| `03_code/scripts/evaluate_noise_robustness.py` | ✅ | ✅ | Synthetic heavy-noise stress test |
| `03_code/scripts/test_shapes.py` | ✅ | ✅ | Tensor shape contract validation |
| `03_code/scripts/compare_against_baseline.py` | ✅ | ✅ | Pass/fail metric gate vs baseline run |

## Pipelines and demo

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `03_code/scripts/run_full_pipeline.py` | ⚠ | ⚠ | End-to-end train → eval → noisy eval → captions |
| `03_code/scripts/live_demo_noisy_folder.py` | ⚠ | ⚠ | Deployment-style demo on arbitrary image folders |
| `03_code/scripts/run_ablation_matrix.py` | ⚠ | ⚠ | Robustness ablations (TTA, postproc, LoRA, etc.) |

## Captioning

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `03_code/scripts/evaluate_captions.py` | — | 🅱 | BERTScore caption quality on test split |
| `03_code/scripts/caption_compare.py` | — | 🅱 | Side-by-side caption variant comparison |
| `03_code/scripts/synthesize_caption_files.py` | — | 🅱 | Build caption reference files from masks |
| `03_code/scripts/run_llm_compression_prompt_tests.py` | 🅰 | 🗄 | Caption INT8 / prompt variant tests |
| `03_code/scripts/aggregate_run22_results.py` | — | 🅱 | Aggregate run22 experiment JSON |
| `03_code/scripts/generate_reports.py` | — | 🅱 | Markdown/CSV report generation from results |

## Compression and distillation

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `03_code/scripts/run_compression_gate_pipeline.py` | 🅰 | 🗄 | Two-stage compression with quality gate |
| `03_code/scripts/run_staged_distillation.py` | 🅰 | 🗄 | Student distillation vs run21 baseline |
| `03_code/scripts/run_head_sanity_tests.py` | 🅰 | 🗄 | Short head-pruning sanity runs |

## Baselines (external models)

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `tools/winclip_eval.py` (from `SSVP/`) | 🅰 | — | **Full** WinCLIP eval on cable/transistor/capsule |
| `03_code/scripts/evaluate_winclip.py` | — | 🅱 | **Stub** — placeholder metrics if repo missing |
| `03_code/scripts/run_anomalyclip_baseline.py` | — | 🅱 | AnomalyCLIP baseline runner |

## Shared utilities

| Script | SSVP | REFAC | Purpose |
|--------|:----:|:-----:|---------|
| `03_code/scripts/path_utils.py` | ✅ | ✅ | Repo-root path constants and import setup |

## External repos (not under `03_code/scripts/`)

| Location | Purpose |
|----------|---------|
| `SSVP/external/WinCLIP/` | WinCLIP baseline implementation |
| `SSVP/external/AnomalyGPT/` | LVLM anomaly detection baseline |
| `SSVP/ssvp/03_code/external/AnomalyCLIP/` | AnomalyCLIP copy for in-tree eval |
| `SSVP/ssvp/03_code/external/WinCLIP/` | Duplicate WinCLIP copy (**consolidate later**) |

## Recommended first commands (canonical SSVP)

```powershell
cd SSVP\ssvp

# 1. Validate install
python 03_code\scripts\test_shapes.py

# 2. Prepare data (after downloading MVTec cable)
python 03_code\scripts\prepare_cable_split.py `
  --source 04_data\datasets\cable `
  --output_root 04_data\datasets\cable_resplit

# 3. Full pipeline
python 03_code\scripts\run_full_pipeline.py `
  --config 03_code\configs\default.yaml `
  --data_root 04_data\datasets\cable_resplit\cable `
  --output_dir 05_results\ablations\my_run `
  --epochs 15

# 4. Live demo
python 03_code\scripts\live_demo_noisy_folder.py `
  --input_folder 04_data\inp\combined `
  --checkpoint 05_results\ablations\run21_resplit_15es\best_model.pth `
  --output_dir 05_results\ablations\live_demo
```

## After merge (planned additions to SSVP)

```powershell
# Caption BERTScore eval (from REFAC)
python 03_code\scripts\evaluate_captions.py --help

# Run22 training overlay
python 03_code\scripts\train_run22.py --config 03_code\configs\run22_override.yaml
```

See [`MERGE_PLAN.md`](MERGE_PLAN.md) for porting order and dependency notes.
