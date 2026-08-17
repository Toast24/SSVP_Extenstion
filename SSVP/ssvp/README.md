# SSVP Submission Repository

Industrial zero-shot anomaly segmentation + captioning on MVTec AD (cable focus).

> **Monorepo note:** This is the **canonical** project inside the [`ssvpall`](../../README.md) workspace. A refactor fork lives at [`../../SSVPREFAC/`](../../SSVPREFAC/README.md) — see [`../../docs/MERGE_PLAN.md`](../../docs/MERGE_PLAN.md) for consolidation status.

## Submission layout

| Folder | Contents |
|--------|----------|
| `01_admin/` | Team info, contribution statements |
| `02_report/` | Final report PDF |
| `03_code/` | Source, configs, scripts |
| `04_data/` | Dataset links, sample inputs, local MVTec copies |
| `05_results/` | Metrics, ablations, figures |
| `06_demo/` | Demo instructions and inputs |
| `07_claims/` | Reproduced vs contributed work |

## Start here

| Task | Document |
|------|----------|
| Run the demo | [`06_demo/demo_instructions.md`](06_demo/demo_instructions.md) |
| Script reference | [`03_code/README.md`](03_code/README.md) |
| Script index (by purpose) | [`03_code/scripts/README.md`](03_code/scripts/README.md) |
| All experiment commands | [`03_code/RUNALLEXPS.md`](03_code/RUNALLEXPS.md) |
| Results and recommendations | [`05_results/RESULTS.md`](05_results/RESULTS.md) |
| Architecture deep-dive | [`../ssvp_technical_walkthrough.md`](../ssvp_technical_walkthrough.md) |
| Install, train, troubleshoot | [`../ssvp_operations_guide.md`](../ssvp_operations_guide.md) |

## Recommended deployment

- Checkpoint: `05_results/ablations/run21_resplit_15es/best_model.pth`
- Entry script: `03_code/scripts/live_demo_noisy_folder.py`
- Caption mode: INT8 text-transformer compression (on by default)
