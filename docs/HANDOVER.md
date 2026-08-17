# Handover — SSVPREFAC merge into canonical SSVP tree

Date: 2026-08-17
Branch: `SSVP_Extension` (single repo now rooted at `ssvpall/`)

## What this covers

This documents the state after merging `SSVPREFAC/` into `SSVP/ssvp/` per
[`docs/MERGE_PLAN.md`](MERGE_PLAN.md), and answers: **is it safe to delete
`SSVPREFAC/` now?**

## Repo/git changes (not in the original plan)

- The git repo was previously rooted at `C:/Users/kedar` (your whole home
  directory) with no commits — not scoped to this project at all. Re-initialized
  a proper repo at `ssvpall/` on a new branch `SSVP_Extension`. The old
  home-directory repo was left untouched (not deleted).
- `SSVP/ssvp/`, `SSVPREFAC/DL_Project_refactor/`, and the vendored
  `external/AnomalyGPT`, `external/WinCLIP`,
  `03_code/external/AnomalyCLIP`, `03_code/external/WinCLIP` each had their
  own nested `.git` folders (all four pointed at
  `github.com/Toast24/DL_Project.git`, i.e. stale/duplicate clones). These
  were removed so everything is tracked by the one `ssvpall` repo. Their
  local commit history is gone; the file contents are not — everything is
  now committed fresh here.
- Two commits so far:
  1. `Initialize SSVP_Extension repo: merge Phase 0-2/4 per docs/MERGE_PLAN.md`
  2. `Recover REFAC-only submission content; port safe train.py diagnostics diff`

## Merge status vs. docs/MERGE_PLAN.md

| Phase | Status |
|---|---|
| 0 — Docs/hygiene | Done (README + `.gitignore` already existed) |
| 1.1 — requirements.txt | Already done (bert-score line present) |
| 1.2 — run22_override.yaml | Done — copied to `SSVP/ssvp/03_code/configs/` |
| 1.3 — path_utils.py | Confirmed identical, no action |
| 2 — Port scripts | Done — see below |
| 3 — Model/training diffs | **Mostly not applicable** — see finding below |
| 4.1 — Caption eval docs | Done — `CAPTION_EVAL_SUMMARY.md`, `caption_eval_aggregated.json` copied; raw per-run JSON/CSV also archived to `05_results/adhoc/run22_raw/` |
| 4.2 — Datasets | Already single-sourced under `SSVP/ssvp/04_data/datasets/` |
| 4.3 — runs/ folder | Small JSON/MD summaries archived; the 12GB `SSVPREFAC/runs/` directory itself was **not** copied (checkpoints/large artifacts, gitignored either way) — see cleanup note below |
| 5 — Archive REFAC | Ready — see verdict below |
| 6 — Doc updates | Done — `03_code/README.md` script index updated |

### Phase 2 — scripts ported into `SSVP/ssvp/03_code/scripts/`

- `evaluate_captions.py`, `caption_compare.py` — import changed to pull
  `_build_caption_target`/`_mask_to_spatial_location` from a new
  **`caption_utils.py`** instead of `run_full_pipeline.py`, so
  `run_full_pipeline.py` itself didn't need to be touched (avoids Phase 3
  GPU-validation risk for an otherwise independent utility).
- `synthesize_caption_files.py`, `generate_reports.py`,
  `aggregate_run22_results.py`, `train_run22.py`,
  `run_anomalyclip_baseline.py` — copied as-is (standalone / subprocess-based,
  no import surgery needed).
- Not ported, per plan: `prepare_category_split.py` (superseded),
  `evaluate_winclip.py` (superseded by `SSVP/tools/winclip_eval.py`).
- `split_dataset.py` reviewed — it's an index-only splitter, functionally
  superseded by `prepare_cable_split.py`'s physical multi-category/nested-dir
  split. No unique logic worth merging.
- Moved to `_deprecated/`: `run_compression_gate_pipeline.py`,
  `run_head_sanity_tests.py`, `run_llm_compression_prompt_tests.py`,
  `run_staged_distillation.py`.

### Phase 3 finding — important

The plan assumed `models/ssvp.py`, `models/losses.py`, `train.py`, etc. had
diverged and needed a careful GPU-validated three-way merge. **On inspection,
they hadn't** — every file under `03_code/src/` was byte-identical between
`SSVP/ssvp` and `SSVPREFAC/DL_Project_refactor` except for CRLF vs LF line
endings (confirmed by diffing with line-ending normalization). The only
*real* differences were in `train.py`:

- REFAC added `log_dataset_statistics()` calls (P-PRO diagnostics: per-category/
  defect sample counts, dumped to `dataset_statistics.json`) — pure logging +
  JSON export, no effect on model math or training loop.
- SSVP already had a small improvement REFAC lacked (`save_periodic_checkpoints`
  config gate).

Ported the diagnostics addition into `SSVP/ssvp/03_code/src/data/mvtec.py`
(new `log_dataset_statistics()` function) and wired it into
`SSVP/ssvp/03_code/scripts/train.py`, keeping SSVP's checkpoint-gate
improvement. This did **not** require GPU validation since it's additive
logging, not a model/loss change — but you should still run
`test_shapes.py` and a short `train.py` smoke run on your GPU box before
trusting it fully, since it hasn't been executed post-merge in this session.

**No `run22_override.yaml` loss-term ("prompt dice") divergence was found** in
`losses.py` — the plan anticipated this but the files were identical.

### Recovered content the plan didn't call out

While diffing directory-by-directory to double check nothing was missed
before recommending SSVPREFAC deletion, found these were **filled-in real
content in SSVPREFAC but still blank templates in SSVP** — now copied over:

- `01_admin/team_info.txt` — real team/mentor names (SSVP had a blank template)
- `01_admin/contribution_statement.pdf` (didn't exist in SSVP)
- `02_report/latex_source.zip` (didn't exist in SSVP)
- `07_claims/prior_work_basis.md` — real literature review content (SSVP had
  a blank template); also removed a stray duplicate file
  `prior_work_basis (1).md` that already had this content under the wrong name
- `06_demo/demo_inputs/*.png` (didn't exist in SSVP)

## Verdict: is it safe to delete `SSVPREFAC/`?

**Yes**, with one caveat. Directory-by-directory diff across
`01_admin`–`07_claims`, `03_code/{scripts,configs,src}`, `04_data`, and
`05_results` found no remaining content unique to `SSVPREFAC` that isn't
now either (a) present in `SSVP/ssvp/`, or (b) intentionally dropped per
the plan (superseded scripts) with that decision recorded above.

**Caveat**: `SSVPREFAC/runs/` is ~12GB (checkpoints + per-image caption
outputs from the run22 experiment). It's already gitignored and its small
summary files (`aggregate_results.json`, `caption_comparison.md`) were
copied to `SSVP/ssvp/05_results/adhoc/`, but the bulk of it (model
checkpoints, hundreds of per-image `.txt` files under
`caption_outputs/by_image/`) was **not** copied anywhere — deleting
`SSVPREFAC/` will permanently lose those specific artifacts. If you don't
need to reproduce/inspect that exact run's raw outputs later, it's fine to
delete. If unsure, keep `SSVPREFAC/runs/` around a bit longer even after
removing the rest of `SSVPREFAC/`.

Also note: `SSVPREFAC/DL_Project_refactor/.git` and the vendored external
repos' `.git` folders were already removed (see git changes above) — so
deleting `SSVPREFAC/` now only loses working-tree files, not any git history
(that history is gone either way, already committed as plain files here).

## Suggested next step (not yet done)

Move `SSVPREFAC/` to `_archive/SSVPREFAC/` (Phase 5's original suggestion)
instead of hard-deleting, if you want a paper trail without keeping it in
the active workspace. Otherwise a straight delete is safe per the verdict
above.
