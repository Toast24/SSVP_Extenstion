# Experiment Summary — Work Done Today

This file summarizes all experiments run today and maps them to the experiment sections described in the post-extension evaluation notes.

**Overview**
- Workspace root: `c:\Users\kedar\SSVP\ssvp`
- Key repositories: `03_code/` (SSVP), `03_code/external/WinCLIP`, `03_code/external/AnomalyCLIP`.

**1. SSVP run21 (Cross-category / sanity check)**
- Purpose: Run the cable-trained `run21` checkpoint on the transistor test set (negative-transfer / cross-category test described in Section E).
- Command used: `03_code/scripts/inference.py` with `--checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth`.
- Results saved: [05_results/ablations/run21_on_transistor/results.json](05_results/ablations/run21_on_transistor/results.json)
- Key metrics from run: image AUROC 43.125; image AP 42.909; pixel AUROC 70.238; pixel PRO 27.563.
- Notes: This is the negative-transfer experiment (E). Use these numbers to report cross-category transfer failure.

**2. WinCLIP Baseline (Experiment D1: Verify WinCLIP)**
- Purpose: Run WinCLIP zero-shot baseline on the 3-category `mvtec_three` split (cable, transistor, capsule).
- Repository: [03_code/external/WinCLIP](03_code/external/WinCLIP)
- Patches applied:
  - [03_code/external/WinCLIP/datasets/mvtec.py](03_code/external/WinCLIP/datasets/mvtec.py) — dataset root/path fix to point to repo `04_data/datasets`.
  - [03_code/external/WinCLIP/WinCLIP/model.py](03_code/external/WinCLIP/WinCLIP/model.py) — ensure model creation receives `img_resize` when applicable.
- Runs completed (240×240 to match backbone):
  - Cable @240: I-AUROC 44.51, P-AUROC 49.41, P-PRO 0.00 — results directory: [05_results/ablations/winclip_d1_cable_res240](05_results/ablations/winclip_d1_cable_res240)
  - Transistor @240: I-AUROC 71.71, P-AUROC 71.03, P-PRO 0.00 — results dir: [05_results/ablations/winclip_d1_transistor_res240](05_results/ablations/winclip_d1_transistor_res240)
  - Capsule @240: I-AUROC 54.57, P-AUROC 84.36, P-PRO 0.00 — results dir: [05_results/ablations/winclip_d1_capsule_res240](05_results/ablations/winclip_d1_capsule_res240)
- Aggregate (mean I-AUROC ≈ 56.93).
- Pending / Notes:
  - Initial attempt at 224×224 failed with positional-embedding mismatch (backbone `ViT-B-16-plus-240` expects 240-grid). Options: run at 240/336/518 (compatible variants), or implement pos-embedding interpolation / change backbone.
  - Remaining resolutions to run: 224 (requires backbone change/interpolation), 336, 518.

**3. AnomalyCLIP Baseline (Experiment D2: trained baseline)**
- Purpose: Run AnomalyCLIP on the same `mvtec_three` split to compare a trained prompt-learning baseline.
- Repository: [03_code/external/AnomalyCLIP](03_code/external/AnomalyCLIP)
- Patches applied:
  - [03_code/external/AnomalyCLIP/dataset.py](03_code/external/AnomalyCLIP/dataset.py) — added `mvtec_three` alias and manifest handling.
  - Generated metadata: `04_data/datasets/mvtec_three_resplit/meta.json`.
- Runs completed:
  - Image-level run (fast path) — final table saved under: [05_results/ablations/anomalyclip_d2_mvtec_three_image](05_results/ablations/anomalyclip_d2_mvtec_three_image)
  - Image-level metrics (per-category): cable I-AUROC 61.3 (I-AP 45.1); transistor I-AUROC 88.2 (I-AP 71.7); capsule I-AUROC 86.4 (I-AP 79.3); mean I-AUROC 78.6.
- Pending / Notes:
  - Pixel-level metrics (pixel AUROC, pixel PRO) require the expensive `cal_pro_score` aggregation step. An initial image-pixel-level run stalled during aggregation in the session; re-run `--metrics image-pixel-level` (or `pixel-level`) and allow the aggregation to finish to obtain P-AUROC and P-PRO.

**4. Implementation & Environment Notes**
- Python venv used: `c:\Users\kedar\SSVP\ssvp\.venv\Scripts\python.exe` (torch 2.11.0+cu126).
- Key scripts used:
  - `03_code/scripts/inference.py` — SSVP inference (run21 checkpoint).
  - `03_code/external/WinCLIP/eval_WinCLIP.py` — WinCLIP evaluation.
  - `03_code/external/AnomalyCLIP/test.py` — AnomalyCLIP evaluation.
- Dependencies installed as needed (e.g., `thop`, `tabulate`) in the venv for AnomalyCLIP.

**5. Where results and artifacts are stored**
- SSVP transistor inference: [05_results/ablations/run21_on_transistor/results.json](05_results/ablations/run21_on_transistor/results.json)
- WinCLIP 240 runs: [05_results/ablations/winclip_d1_*_res240](05_results/ablations/)
- AnomalyCLIP image-level run: [05_results/ablations/anomalyclip_d2_mvtec_three_image](05_results/ablations/anomalyclip_d2_mvtec_three_image)
- Patches applied in source: [03_code/external/WinCLIP/datasets/mvtec.py](03_code/external/WinCLIP/datasets/mvtec.py), [03_code/external/WinCLIP/WinCLIP/model.py](03_code/external/WinCLIP/WinCLIP/model.py), [03_code/external/AnomalyCLIP/dataset.py](03_code/external/AnomalyCLIP/dataset.py).

**6. Immediate conclusions (for paper sections)**
- Section D1 (WinCLIP Baseline): Current 240×240 WinCLIP runs produce mean I-AUROC ≈ 56.9; cable result is anomalously low (≈44.5) and must be investigated. Re-running at 336/518 (and resolving 224) is required before finalizing Table A in the paper.
- Section D2 (AnomalyCLIP): Image-level results are complete (mean I-AUROC 78.6) and provide a trained-baseline comparison. Pixel-level A-PRO/P-AUROC remain pending and are required to compute pixel metrics for Table A/B.
- Section E (Cross-category negative test): SSVP `run21` on transistor (cross-category) shows low image AP and AUROC in this particular run — include this negative-transfer result as-is (it supports the claim that SSVP is not category-agnostic).

**7. Next actions (recommended, high priority)**
1. Re-run AnomalyCLIP with `--metrics image-pixel-level` and wait for `cal_pro_score` aggregation to finish (one job; expected runtime longer). Save outputs under `05_results/ablations/anomalyclip_d2_mvtec_three_pixel`.
2. Run WinCLIP for resolutions 336 and 518 (and attempt 224 after resolving pos-embed/backbone). Use the same experiment commands used for 240 with `--img-resize`/`--img-cropsize` set appropriately.
3. Once above complete, regenerate the paper result tables (clean & noisy) and update `07_claims/post_extension_evaluation.md` to include corrected baseline numbers.

---

Generated on: 2026-05-29

## Update: AnomalyCLIP pixel-level results (2026-05-29)

- Pixel-level metrics completed for `mvtec_three` (saved under `05_results/ablations/anomalyclip_d2_mvtec_three_pixel`).
- Per-category pixel metrics (from the run):
  - cable: pixel AUROC 86.1, pixel PRO 64.7
  - transistor: pixel AUROC 70.7, pixel PRO 63.4
  - capsule: pixel AUROC 98.9, pixel PRO 94.0
  - mean: pixel AUROC 85.2, pixel PRO 74.0

Notes: Image-level metrics (I-AUROC/I-AP) match the earlier image-only run: cable 61.3/45.1, transistor 88.2/71.7, capsule 86.4/79.3.

## Update: WinCLIP 336×336 results (2026-05-29)

- Cable @336: I-AUROC 51.46, P-AUROC 48.51, P-PRO 0.00 — results dir: `05_results/ablations/winclip_d1_cable_res336`.

- Transistor @336: I-AUROC 71.96, P-AUROC 63.54, P-PRO 0.00 — results dir: `05_results/ablations/winclip_d1_transistor_res336`.

- Capsule @336: I-AUROC 57.72, P-AUROC 82.83, P-PRO 0.00 — results dir: `05_results/ablations/winclip_d1_capsule_res336`.

## Update: WinCLIP 518×518 results (2026-05-29)

- Cable @518: I-AUROC 55.55, P-AUROC 52.63, P-PRO 0.00 — results dir: `05_results/ablations/winclip_d1_cable_res518`.

- Transistor @518: I-AUROC 57.21, P-AUROC 63.00, P-PRO 0.00 — results dir: `05_results/ablations/winclip_d1_transistor_res518`.

- Capsule @518: I-AUROC 56.16, P-AUROC 78.64, P-PRO 0.00 — results dir: `05_results/ablations/winclip_d1_capsule_res518`.





