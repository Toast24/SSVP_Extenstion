# SSVPREFAC — Development / Refactor Workspace

This folder is a **working copy** of the SSVP project used for caption evaluation (run22) and script cleanup. It is **not** the canonical submission tree.

## Use the canonical project instead

For demos, baselines, datasets, and submission:

→ **[`../SSVP/ssvp/`](../SSVP/ssvp/)**

Workspace overview:

→ **[`../README.md`](../README.md)**

## What lives here

| Path | Role |
|------|------|
| `DL_Project_refactor/` | Same 01–07 layout as `SSVP/ssvp/` with REFAC-specific scripts |
| `gpu_venv/`, `.venv/` | Local Python environments — **do not commit** |
| `runs/` | Ad-hoc experiment outputs |

## Unique value in this tree (to be merged upstream)

Scripts not yet in `SSVP/ssvp/03_code/scripts/`:

- `evaluate_captions.py` — BERTScore caption evaluation
- `caption_compare.py` — caption variant comparison
- `train_run22.py` + `configs/run22_override.yaml` — prompt-dice experiment
- `generate_reports.py`, `aggregate_run22_results.py`
- `synthesize_caption_files.py`

Deprecated here (moved to `scripts/_deprecated/`), still active in SSVP:

- Compression gate, staged distillation, LLM compression tests, head sanity tests

## Merge status

See [`../docs/MERGE_PLAN.md`](../docs/MERGE_PLAN.md). After merge completes, this folder will be archived under `_archive/SSVPREFAC/`.

## Quick run (this tree only)

```powershell
cd DL_Project_refactor
..\gpu_venv\Scripts\Activate.ps1   # or your own venv
python 03_code\scripts\test_shapes.py
```

Full commands: `DL_Project_refactor/03_code/README.md`
