# Merge and Refactor Plan

Goal: one canonical tree under `SSVP/ssvp/` with REFAC improvements ported in, clear docs, and no duplicate venvs or stub scripts.

## Principles

1. **Canonical root:** `SSVP/ssvp/` — keep submission layout, datasets, baselines, and top-level guides.
2. **Archive, don't delete:** move `SSVPREFAC/` to `_archive/SSVPREFAC/` after merge (or keep with a pointer README).
3. **One venv:** `.venv` at `SSVP/ssvp/` only; gitignore `gpu_venv/`, `SSVPREFAC/.venv/`.
4. **One WinCLIP path:** keep `SSVP/tools/winclip_eval.py` + `SSVP/external/WinCLIP`; drop REFAC stub.
5. **Script namespaces:** active scripts in `03_code/scripts/`; retired scripts in `03_code/scripts/_deprecated/`.

## Phase 0 — Documentation and hygiene (done / low risk)

- [x] Root [`README.md`](../README.md) and `docs/` guides
- [x] Root [`.gitignore`](../.gitignore) for venvs and `runs/`
- [ ] Add `SSVPREFAC/README.md` marking it as staging until Phase 3

## Phase 1 — Requirements and path unification

**1.1 Merge `requirements.txt`**

Into `SSVP/ssvp/03_code/requirements.txt`, add optional caption-eval deps:

```text
bert-score>=0.3.13   # caption quality evaluation (from REFAC)
```

**1.2 Single config baseline**

- Diff `configs/default.yaml` between trees; pick SSVP as base.
- Add `configs/run22_override.yaml` from REFAC as an overlay config (already exists only in REFAC).

**1.3 Verify `path_utils.py`**

Already identical — no change needed.

## Phase 2 — Port REFAC scripts into SSVP

Copy these from `SSVPREFAC/DL_Project_refactor/03_code/scripts/` → `SSVP/ssvp/03_code/scripts/`:

| Script | Action | Dependency notes |
|--------|--------|------------------|
| `evaluate_captions.py` | **Port** | Requires `_build_caption_target`, `_mask_to_spatial_location` from REFAC `run_full_pipeline.py` — extract to `scripts/caption_utils.py` or merge into SSVP `run_full_pipeline.py` |
| `caption_compare.py` | **Port** | Depends on caption pipeline |
| `synthesize_caption_files.py` | **Port** | Standalone |
| `generate_reports.py` | **Port** | Aggregates JSON under `05_results/` |
| `aggregate_run22_results.py` | **Port** | Run22-specific; rename to `aggregate_experiment_results.py` if generalized |
| `train_run22.py` | **Port** | Pair with `configs/run22_override.yaml` |
| `run_anomalyclip_baseline.py` | **Port** | Wire to existing `03_code/external/AnomalyCLIP` |
| `prepare_category_split.py` | **Skip** | Superseded by SSVP `prepare_cable_split.py` (multi-category + nested dirs) |
| `split_dataset.py` | **Review** | Merge logic into `prepare_cable_split.py` if unique |
| `evaluate_winclip.py` | **Drop** | Replace with `SSVP/tools/winclip_eval.py` |

Move from SSVP active → `_deprecated/` (matching REFAC cleanup):

- `run_compression_gate_pipeline.py`
- `run_head_sanity_tests.py`
- `run_llm_compression_prompt_tests.py`
- `run_staged_distillation.py`

Keep them available but out of the main README script list.

## Phase 3 — Merge model / training diffs

Order matters — do this in a GPU environment with `test_shapes.py` passing after each step.

**3.1 `run_full_pipeline.py`**

Merge REFAC caption helpers:

```python
_mask_to_spatial_location()
_build_caption_target()
```

into SSVP version without breaking existing demo flags.

**3.2 `models/ssvp.py` and `models/losses.py`**

Three-way review:

- Keep SSVP distillation/pruning hooks unless REFAC explicitly removed dead code safely.
- Bring in run22 loss terms (prompt dice) behind config flags.

**3.3 `train.py`**

Unify:

- REFAC `train_all_types`, `use_anomaly_supervision` flags
- SSVP early-stopping defaults
- Expose via `default.yaml` rather than forked train scripts long-term

End state: one `train.py` + optional config overlays (`run22_override.yaml`).

## Phase 4 — Results and data consolidation

**4.1 Caption eval artifacts**

Copy unique REFAC results docs (not large checkpoints):

- `05_results/CAPTION_EVAL_SUMMARY.md`
- `05_results/caption_eval_aggregated.json`

into `SSVP/ssvp/05_results/` if not already present.

**4.2 Datasets**

Keep datasets only under `SSVP/ssvp/04_data/datasets/`. REFAC should reference via docs, not duplicate.

**4.3 `runs/` folder**

Move `SSVPREFAC/runs/` contents into `SSVP/ssvp/05_results/adhoc/` or delete after extracting useful JSON/MD.

## Phase 5 — Final layout (target)

```
ssvpall/
├── README.md
├── docs/
├── SSVP/
│   ├── ssvp/                    # ← single canonical project
│   ├── external/
│   ├── tools/
│   ├── results/
│   └── *.md guides
└── _archive/
    └── SSVPREFAC/               # frozen snapshot post-merge
```

## Phase 6 — Documentation updates after merge

1. Update `SSVP/ssvp/03_code/README.md` script list (caption eval + run22).
2. Update `RUNALLEXPS.md` with caption BERTScore and run22 sections.
3. Add `03_code/scripts/README.md` quick index grouped by purpose:
   - **Data prep** — `prepare_cable_split.py`
   - **Train / eval** — `train.py`, `inference.py`, `evaluate_noise_robustness.py`
   - **Pipeline** — `run_full_pipeline.py`, `live_demo_noisy_folder.py`
   - **Caption** — `evaluate_captions.py`, `caption_compare.py`
   - **Baselines** — `../../tools/winclip_eval.py`, `run_anomalyclip_baseline.py`
   - **Deprecated** — `_deprecated/*`

## Risk checklist

| Risk | Mitigation |
|------|------------|
| Broken imports after merge | Run `test_shapes.py` + one `inference.py` smoke test |
| Config drift | Single `default.yaml`; overrides in named YAML files only |
| Duplicate WinCLIP copies | Keep `SSVP/external/WinCLIP`; remove nested duplicate in `03_code/external/` later |
| Huge git repo | Gitignore `*.pth`, `04_data/datasets/`, venvs, `runs/` |
| REFAC-only checkpoint paths | Search-replace `05_results/ablations/run22*` paths in ported scripts |

## Suggested merge order (one PR per phase)

1. Docs + gitignore (Phase 0)
2. Requirements + run22 config (Phase 1)
3. Caption utils extraction + evaluate_captions port (Phase 2–3.1)
4. Model/loss/train unification (Phase 3.2–3.3) — **needs GPU validation**
5. Archive REFAC (Phase 5)
