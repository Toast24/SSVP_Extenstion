# SSVP Post-Extension Evaluation — Program-Committee-Grade Stress Test

---

## A. Sanity-Check on Workshop-Readiness (150 words max)

**Upgraded from "below the bar" to "at the bar, conditionally."** The extension addressed the single biggest gap flagged in [pre_extensioneval.md](file:///home/vivek/DL%20Project/SSVP_Extenstion/07_claims/pre_extensioneval.md): single-category evaluation. You now have three MVTec AD categories (cable, transistor, capsule) with clean *and* noisy results, plus a head-to-head WinCLIP zero-shot baseline on the same categories. Your SSVP mean I-AUROC of **95.09%** (clean, 3-cat mean) vs WinCLIP's **56.89%** is a 38-point gap — a legitimate headline number. However, two things prevent a clean "yes": (1) your `claimed_contribution.md` and `prior_work_basis.md` are **completely empty** — every bullet is a dash with no text; (2) transistor AP is anomalously low (43.82%) despite high I-AUROC (93.05%), which a reviewer will question. **Strongest selling point to lead with**: 38 pp I-AUROC advantage over WinCLIP zero-shot across three categories with no category-specific tuning.

---

## B. Venue Fit (150 words max)

Your paper is a **reproducibility + extension study** of a concurrent preprint (SSVP, arXiv:2601.09147), with WinCLIP benchmarking and a noise robustness characterization. It is not a new method. That profile fits:

1. **`[P0]` CVPR 2027 Workshop on Visual Anomalies in the Wild (VAND)** — Ran at CVPR 2023, 2024. Confident it runs annually. Deadline typically March 2027. Best fit for anomaly detection extension + benchmarking papers. This is your safest primary target.

2. **`[P1]` NeurIPS 2026 Workshop on Distribution Shifts** — Ran at NeurIPS 2023, 2024. Less certain it runs in 2026 — verify at neurips.cc. Your noise robustness angle fits the distribution shift framing.

3. **`[P1]` ECCV 2026 Workshop on Anomaly Detection (ROAD+/VAND)** — ECCV 2026 workshop deadline is typically July–August 2026. Verify at eccv2026.ecva.net. Tight but feasible if you start now.

Your primary target of "NeurIPS 2026 workshop on Zero Shot Anomaly Detection" — I am not aware of a workshop with this exact title. The VAND/ROAD+ community at CVPR/ECCV is where ZSAD papers land. I would target ECCV 2026 as primary (July–August deadline) and CVPR 2027 VAND as fallback.

---

## C. The 3–5 Gaps That Will Cause a Desk-Reject or Reviewer-Reject

### Gap 1 — WinCLIP Numbers Are Anomalously Low vs. Published Results `[P0]`

> *"The WinCLIP numbers reported here (56.89% mean I-AUROC across cable/transistor/capsule) are drastically below the 88.7% MVTec-AD I-AUROC reported in the original WinCLIP paper (Jeong et al., CVPR 2023) and the 91.8% cited in your own Table 4. Either the WinCLIP implementation is broken, the preprocessing pipeline is incorrect, or the comparison is unfair. A 38-point advantage against a broken baseline is meaningless."*

From your [CLEAN_NOISY_EVAL_SUMMARY.md](file:///home/vivek/DL%20Project/SSVP_Extenstion/05_results/ablations/CLEAN_NOISY_EVAL_SUMMARY.md), line 29: the WinCLIP run used `resize 240, center-crop 240` — this is a **240×240** resolution, while the original WinCLIP paper uses **240×240** for some backbones but the standard CLIP ViT-L/14 expects **224×224** or **336×336**. More critically, WinCLIP cable I-AUROC of **44.64%** is below random chance — something is fundamentally wrong with this baseline.

**Minimum evidence to defuse**:

- Verify the WinCLIP checkpoint, backbone version (ViT-B/16+? ViT-L/14?), and resolution match the original paper's configuration.
- Report which WinCLIP codebase you used (official Jeong et al. release or a third-party reimplementation). The code availability status of the *official* WinCLIP is uncertain — I am not sure the original authors released public code. Verify on GitHub.
- If the WinCLIP implementation is the anomalyCLIP community's re-implementation, state that explicitly.
- Re-run WinCLIP with the *original paper's* recommended hyperparameters (window sizes [2,3], proper resolution) and report the results. If WinCLIP still underperforms, add a paragraph in the paper explaining why (likely: the 70/20/10 anti-leakage split means fewer test samples, and WinCLIP's zero-shot compositional prompts are not tuned for this partition).

---

### Gap 2 — SSVP Is Trained, WinCLIP Is Zero-Shot — Unfair Comparison `[P0]`

> *"SSVP is trained for 15 epochs on the normal samples of each category. WinCLIP is run in zero-shot mode (k-shot = 0). Comparing a trained model to a truly zero-shot baseline and claiming a 38-point advantage is misleading. This is not an apples-to-apples comparison."*

Your report's abstract says "zero-shot industrial anomaly detection" but your SSVP pipeline trains on normal samples with supervision (15 epochs, early stopping, threshold calibration on a validation split). This is technically **few-shot** or **one-class classification**, not zero-shot. WinCLIP at k=0 is genuinely zero-shot.

**Minimum evidence to defuse**:

- Reframe the comparison honestly: "SSVP (trained on normal samples) vs WinCLIP (zero-shot)" — make the training regime difference explicit in every table caption and in the abstract.
- Ideally, also benchmark AnomalyCLIP (Zhou et al., 2024, arXiv:2310.18961; code public at `github.com/zqhang/AnomalyCLIP`) under the same protocol to have a second trained-baseline comparison. Estimated compute: ~1 GPU-hour per category inference-only.

---

### Gap 3 — Transistor AP Anomaly (43.82%) `[P1]`

> *"Transistor achieves 93.05% I-AUROC but only 43.82% I-AP. This means the model's ranking is reasonable but its confidence calibration is severely off — it assigns high anomaly scores to many normal images. The paper does not acknowledge or analyze this discrepancy."*

From [all_results_combined.json](file:///home/vivek/DL%20Project/SSVP_Extenstion/05_results/ablations/all_results_combined.json), transistor clean: `auroc: 93.05, ap: 43.82`. This is a 49-point gap between AUROC and AP. Compare cable (97.42 AUROC, 93.62 AP — 4-point gap) and capsule (94.79 AUROC, 90.25 AP — 4.5-point gap). Transistor is a clear outlier.

**Minimum evidence to defuse**: Add a paragraph analyzing this — likely cause is extreme class imbalance (only 8 anomalous out of 63 test samples for transistor, per `n_anomalous: 8`). When anomaly prevalence is ~12.7%, AP is highly sensitive to false positives. Report the precision-recall curve or at minimum the number of false positives at the F1-optimal threshold.

---

### Gap 4 — P-PRO Inconsistency Across Categories `[P1]`

> *"Cable P-PRO is 29.54%, transistor P-PRO is 75.71%, capsule P-PRO is 88.60%. The original SSVP paper reports MVTec-AD mean P-PRO of 89.0%. The cable P-PRO collapse was flagged in the original evaluation but remains unexplained."*

From pre_extensioneval.md line 31: *"Run21 achieves 95.34% pixel-AUROC but only 29.54 P-PRO."* The new multi-category data shows this is cable-specific — transistor and capsule have healthy P-PRO. This actually helps your paper: it suggests the 70/20/10 anti-leakage split specifically hurts cable's PRO threshold calibration, not the model architecture.

**Minimum evidence to defuse**: Add analysis showing P-PRO is threshold-sensitive and your anti-leakage validation set for cable may yield a suboptimal per-region threshold. Show what happens if you use a different PRO threshold (e.g., the one that maximizes PRO on validation).

---

## D. The 1–2 Critical New Experiments

### Experiment D1 `[P0]` — Verify WinCLIP Baseline Is Not Broken

- **Dataset**: MVTec AD cable, transistor, capsule (same three categories you already have).
- **Model**: WinCLIP (Jeong et al., 2023, CVPR; code availability uncertain — verify whether the implementation at `github.com/zqhang/AnomalyCLIP` includes a WinCLIP reproduction or whether you used `github.com/caoyunkang/WinCLIP` — I am not certain either is the official release).
- **Baseline**: Your current WinCLIP run uses `resize 240, center-crop 240`. The original WinCLIP paper evaluates at the native CLIP resolution. Re-run with:
  - `resize 518` (matching your SSVP input resolution) **OR** `resize 224` (matching CLIP ViT-B/16 native resolution), depending on which backbone your WinCLIP implementation uses.
  - Window scales: `[2, 3]` (same as current).
  - Seed: `42`.
- **Metric**: I-AUROC, P-AUROC, P-PRO on clean test split.
- **Hyperparameter range**: Only the input resolution. Try {224, 240, 336, 518}.
- **Estimated compute**: ~2 GPU-hours total (inference-only, four resolution variants × three categories).
- **WIN**: WinCLIP clean cable I-AUROC rises to ≥75% (closer to the published ~85–91% range for MVTec AD), and your SSVP advantage narrows but remains ≥15 pp. This makes your comparison credible.
- **FORCES REFRAME**: If WinCLIP at correct resolution matches or exceeds 90% I-AUROC on your split (close to published numbers), your SSVP advantage shrinks to <5 pp and the headline number is no longer a contribution. Reframe to the noise robustness + captioning angle (Section H).

---

### Experiment D2 `[P1]` — AnomalyCLIP Baseline Under Your Protocol

- **Dataset**: MVTec AD cable, transistor, capsule — your 70/20/10 anti-leakage split.
- **Model**: AnomalyCLIP (Zhou et al., 2024, arXiv:2310.18961; code is public at `github.com/zqhang/AnomalyCLIP`). Run in zero-shot mode with the released checkpoint.
- **Metric**: I-AUROC, P-AUROC, P-PRO on clean test split.
- **Hyperparameter range**: None — use default AnomalyCLIP config.
- **Estimated compute**: ~1.5 GPU-hours (inference-only, three categories).
- **WIN**: SSVP I-AUROC exceeds AnomalyCLIP by ≥5 pp on at least 2/3 categories. This gives you a second competitive baseline and strengthens the paper's credibility since AnomalyCLIP is also a trained model (prompt learning).
- **FORCES REFRAME**: If AnomalyCLIP matches or exceeds SSVP on 2/3 categories, the SSVP contribution is diminished — pivot to the noise robustness angle where you have unique data.

---

## E. Negative-Result Experiment to Run Anyway

### `[P1]` — Cross-Category Prompt Transfer Failure

Run SSVP inference on the **transistor** test set using the **cable-trained** checkpoint (run21) — i.e., test cross-category zero-shot transfer without retraining. Report I-AUROC and P-AUROC.

- **Dataset**: MVTec AD transistor test split (your 70/20/10 partition).
- **Model**: Cable run21 checkpoint, no retraining.
- **Metric**: I-AUROC, P-AUROC.
- **Estimated compute**: ~0.5 GPU-hours (inference-only).
- **Expected result**: I-AUROC will likely collapse to 50–65% because the VCPG has learned cable-specific prompt distributions. This is the honest negative result.
- **Why it strengthens the paper**: It demonstrates that SSVP's per-category training is load-bearing — the model does *not* actually achieve zero-shot generalization across categories. This is a valuable finding that separates your paper from overclaiming. A reviewer who sees you voluntarily reporting this will trust your other numbers more.
- **If it succeeds (I-AUROC ≥ 80%)**: That would actually be a positive surprise and a genuine contribution — report it as evidence that SSVP's learned prompt space partially transfers across MVTec categories.

---

## F. Week-by-Week Plan (4 weeks, ~20 h/week, team of 2)

### Week 1: Baseline Verification and Critical Fixes

| | Task |
|---|---|
| **Compute** | (i) Re-run WinCLIP at correct resolution (D1) — assign team member 1. (ii) Run AnomalyCLIP inference on 3 categories (D2) — assign team member 2. Both tasks are parallelizable. |
| **Writing** | Fill in `claimed_contribution.md` and `prior_work_basis.md` — **today, before any compute work**. Rewrite abstract to explicitly state "SSVP (trained) vs WinCLIP (zero-shot)" comparison framing. |
| **Decision gate** | If corrected WinCLIP cable I-AUROC rises above 85% → your 38-point headline number collapses. Recalculate the true delta. If true delta < 10 pp → pivot to Section H immediately. If WinCLIP stays below 70% even at correct resolution → your current comparison is defensible but you **must** explain why your WinCLIP numbers diverge from published ones. |

### Week 2: Negative Result + Paper Rewrite Core

| | Task |
|---|---|
| **Compute** | (i) Cross-category transfer experiment (E). (ii) Run INT8 caption evaluation on 20+ images with BERTScore (from pre_extensioneval.md Section E recommendation). |
| **Writing** | (i) Rewrite Section 5 (Experiments) with a clean 3-category results table: SSVP vs WinCLIP (and optionally AnomalyCLIP). (ii) Write transistor AP analysis paragraph (Gap 4). (iii) Write P-PRO cable analysis paragraph (Gap 5). |
| **Decision gate** | Are your corrected baselines + multi-category results telling a coherent story? If SSVP dominates on 3/3 categories with corrected baselines → proceed to workshop submission path. If SSVP loses on 1+ category → narrow claims to per-category analysis. |

### Week 3: Full Paper Draft

| | Task |
|---|---|
| **Compute** | Buffer — re-run any experiments that produced suspicious results in weeks 1–2. |
| **Writing** | (i) Rewrite Intro, Related Work, Method (see Section G below). (ii) Write Limitations section covering: unfair training comparison, 3-category scope, synthetic noise only, cable P-PRO anomaly. (iii) Write Conclusion with the cross-category transfer negative result. |
| **Decision gate** | Is the paper ≥6 pages of substantive content (4-page workshop format or 6–8 page extended abstract, depending on venue)? If under 4 pages → arXiv-only. Have one team member act as hostile reviewer. |

### Week 4: Polish and Submit

| | Task |
|---|---|
| **Compute** | None. |
| **Writing** | (i) Full paper internal review. (ii) Check every number against JSON files — one inconsistency will sink a desk review. (iii) Verify all arXiv IDs for references [1], [8], [12], [13], [16], [17]. (iv) Format for target venue template. |
| **Decision gate** | Submit to ECCV 2026 workshop if deadline is open. If not, submit to arXiv and target CVPR 2027 VAND. |

---

## G. Section-by-Section Paper Revision Plan

### Title — **Revise heavily**

Current: *"SSVP: Robust Industrial Anomaly Segmentation and Captioning"* — implies you are the SSVP authors.

New skeleton:

- "Beyond Single-Backbone ZSAD: A Multi-Category Evaluation of SSVP with Noise Robustness Benchmarking"
- Or: "Evaluating SSVP for Zero-Shot Anomaly Detection: Multi-Category Benchmarking Against WinCLIP with Noise Robustness Analysis"

> [!WARNING]
> The current title will make a reviewer think you are claiming SSVP as your own invention. This alone can trigger a plagiarism flag at desk review.

---

### Abstract — **Rewrite from scratch**

Skeleton:

1. One sentence: Zero-shot AD is constrained by single-backbone architectures and lacks robustness benchmarking.
2. One sentence: We evaluate SSVP (Fu et al., 2026), a dual-backbone CLIP+DINOv2 framework, on three MVTec AD categories and benchmark against WinCLIP zero-shot.
3. Headline number: SSVP achieves 95.09% mean I-AUROC across cable/transistor/capsule vs WinCLIP's [corrected number from D1].
4. Noise finding: Under synthetic Gaussian noise, SSVP degrades by [X] pp while maintaining [Y]% I-AUROC.
5. Negative finding: Backbone compression via depth and differentiated pruning fails the quality gate, confirming the architecture is not straightforwardly compressible.

> [!CAUTION]
> Do **not** use the phrase "state of the art" anywhere. You have 3 categories of 1 dataset. You have not evaluated on VisA, BTAD, KSDD2, RSDD, DAGM, or DTD-Syn.

---

### Introduction — **Revise heavily**

Skeleton:

1. Industrial AD needs both accuracy and noise robustness — cite MVTec AD (Bergmann et al., CVPR 2019) as the standard benchmark.
2. ZSAD via VLMs (CLIP) is promising but under-benchmarked for robustness — cite WinCLIP (Jeong et al., CVPR 2023), AnomalyCLIP (Zhou et al., 2024).
3. SSVP (Fu et al., arXiv 2026) proposes dual-backbone fusion — **acknowledge it is a concurrent preprint**.
4. Our contributions: (a) multi-category evaluation of SSVP on 3 MVTec categories; (b) head-to-head WinCLIP benchmarking under identical protocol; (c) noise robustness characterization; (d) captioning branch with INT8 compression.
5. Scope statement: single GPU, 3 categories, synthetic noise only.

> [!IMPORTANT]
> Current intro (report line 67): *"The SSVP framework represents the state of the art in this paradigm"* — a reviewer will read this as you claiming SOTA for your extension. Remove or qualify: "SSVP reports competitive results on 7 datasets in its preprint."

---

### Related Work — **Keep as-is, add one paragraph**

Add a **"Reproducibility and Benchmarking Studies"** paragraph acknowledging that your work is positioned as an evaluation study, not a new method. Cite ML Reproducibility Challenge papers as framing precedent. This pre-empts the "this is just an implementation" objection.

---

### Method — **Revise lightly**

The HSVS/VCPG/VTAM descriptions in `ssvp.py`, `vcpg.py`, `vtam.py` are clean and well-documented. Keep the method section largely intact. 
Add:

- Exact image counts per split per category (cable: 74 test / transistor: 63 test / capsule: 70 test — from your JSONs).
- Explicit statement of the training regime difference: "SSVP trains on normal images; WinCLIP uses no training data."

---

### Experiments — **Rewrite from scratch**

Current Table 3 (report line 389) is a run ledger with 14 experiments in a single table. Replace with:

**Table A: Multi-Category Clean Results — SSVP vs WinCLIP**

| Category | Model | I-AUROC | I-F1 | I-AP | P-AUROC | P-PRO | P-AP |
|---|---|---:|---:|---:|---:|---:|---:|
| cable | SSVP | 97.42 | 84.85 | 93.62 | 95.34 | 29.54 | 50.89 |
| cable | WinCLIP | 44.64 | 76.35 | 56.97 | 49.43 | 29.41 | 3.02 |
| transistor | SSVP | 93.05 | 90.00 | 43.82 | 94.75 | 75.71 | 59.98 |
| transistor | WinCLIP | 71.71 | 67.39 | 56.00 | 71.00 | 0.00 | 12.47 |
| capsule | SSVP | 94.79 | 82.35 | 90.25 | 97.75 | 88.60 | 47.56 |
| capsule | WinCLIP | 54.33 | 90.46 | 85.42 | 84.39 | 0.00 | 5.76 |
| **mean** | **SSVP** | **95.09** | **85.73** | **75.90** | **95.94** | **64.62** | **52.81** |
| **mean** | **WinCLIP** | **56.89** | **78.07** | **66.13** | **68.27** | **9.80** | **7.08** |

> [!WARNING]
> The WinCLIP cable I-AUROC of 44.64% is below random. This **will** be the first thing a reviewer looks at. You must resolve D1 before publishing this table. If WinCLIP is genuinely broken in your implementation, this table makes the paper look dishonest, not strong.

**Table B: Noise Robustness (Clean → Noisy Delta)**

| Category | Model | ΔI-AUROC | ΔP-AUROC |
|---|---|---:|---:|
| cable | SSVP | −10.42 | −3.52 |
| cable | WinCLIP | +1.78 | −3.61 |
| transistor | SSVP | −11.91 | −23.30 |
| transistor | WinCLIP | −29.29 | −8.52 |
| capsule | SSVP | −14.34 | −4.97 |
| capsule | WinCLIP | +2.15 | −8.26 |

Move the current 14-experiment run ledger (Table 3) to an appendix.

---

### Ablations — **Revise heavily**

Lead with the consistency regularization result: *"Removing consistency regularization collapses noisy I-AUROC from 87.00% to 42.76% — below random"* (report line 357). This is the single most interesting mechanistic finding in the paper.

Skeleton:

1. Consistency regularization is load-bearing (no_consistency ablation)
2. LoRA does not compensate (lora_no_consistency)
3. Compression gate: both pruning strategies fail
4. DAE: helps noise tolerance locally but hurts global discrimination
5. Negative results section: PromptDice, robust patch, entropy calibration — all fail

> [!NOTE]
> The consistency regularization result from your ablation at `ablation/no_consistency` (clean I-AUROC drops from 97.42→78.67, noisy I-AUROC drops to 42.76) is genuinely publishable on its own as a mechanistic finding. Emphasize it.

---

### Limitations — **Keep structure, expand**

Add:

1. `[P0]` SSVP is trained on normal samples; WinCLIP is truly zero-shot — the comparison is not apples-to-apples.
2. `[P0]` WinCLIP numbers diverge from published results — state why (resolution mismatch, split difference, or implementation issue).
3. `[P1]` Only 3 of 15 MVTec categories evaluated; no VisA, BTAD, KSDD2 evaluation.
4. `[P1]` Captioning evaluation remains limited (Table 2 is 3 images).
5. `[P1]` SSVP [1] is an unreviewed preprint.
6. `[P2]` Synthetic noise only — no real sensor noise profiles.

---

### Conclusion — **Rewrite from scratch**

One paragraph: what you showed (multi-category SSVP dominance over WinCLIP zero-shot, with caveats about training regime), what fails (compression, DAE, PromptDice), what is deployable (run21 + INT8 captioning), and the key open question (cross-category transfer — if you run Experiment E).

---

## H. Backup Plan if D1 Reveals a Broken WinCLIP Baseline

### Scenario 1: Corrected WinCLIP achieves ≥85% I-AUROC (matching published numbers)

Your SSVP advantage shrinks from 38 pp to ~10 pp. This is still a positive result but changes the paper's story.

**Reframe**: "SSVP provides modest but consistent improvements over WinCLIP (+10 pp I-AUROC) while additionally offering noise robustness characterization and a captioning pipeline. The contribution is the evaluation methodology and robustness benchmarking, not the magnitude of improvement."

**Venue**: Still submittable to CVPR 2027 VAND or ECCV 2026 anomaly workshop — benchmarking papers are valued.

### Scenario 2: WinCLIP cannot be fixed (implementation is genuinely broken or official code is unavailable)

**Reframe**: Drop WinCLIP from the comparison entirely. Reframe as "A Multi-Category Reproducibility Study of SSVP with Noise Robustness Benchmarking." Compare only to the SSVP paper's own reported numbers (your Table 4). Your contribution becomes: reproducing SSVP on 3 categories, characterizing noise degradation, and showing that compression strategies fail.

**Venue**: arXiv tech report (cs.CV) + CVPR 2027 VAND. This is honest and still citable.

### Scenario 3: Both D1 and D2 fail (corrected baselines match or exceed SSVP)

**Reframe**: "A Negative Reproducibility Report on SSVP: When Do Multi-Backbone Architectures Actually Help?" This is a valid and valuable contribution — negative reproducibility results are publishable at MLRC (ML Reproducibility Challenge) and NeurIPS Workshops on Negative Results. The consistency regularization finding (Section 6 ablation) becomes the headline.

**Venue**: ML Reproducibility Challenge (typically NeurIPS-affiliated, deadline around September–October). arXiv simultaneously.

---

## Final Verdict (≤80 words)

Your project has improved materially since the pre-extension evaluation. The multi-category extension and WinCLIP benchmarking were the right moves. However, the WinCLIP cable I-AUROC of 44.64% is a **red flag** — if your baseline is broken, the entire headline number collapses. Additionally, both `claimed_contribution.md` and `prior_work_basis.md` are empty. If you fix the WinCLIP baseline (D1) and it still shows a ≥15 pp SSVP advantage, I estimate **50–55% acceptance probability** at an appropriate workshop after 4 weeks of focused work. If WinCLIP turns out to be broken and unfixable, **aim for arXiv-only this cycle**.
