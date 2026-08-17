# SSVP Workspace

Industrial zero-shot anomaly detection and segmentation on MVTec AD, extended with noise robustness, defect captioning, and model compression experiments.

This workspace contains **two related codebases** that share the same submission layout (`01_admin` … `07_claims`) but diverged during development. Use this file as the entry point.

## Quick start

| Goal | Start here |
|------|------------|
| Run the recommended demo (run21 + captions) | [`SSVP/ssvp/06_demo/demo_instructions.md`](SSVP/ssvp/06_demo/demo_instructions.md) |
| Understand architecture | [`SSVP/ssvp_technical_walkthrough.md`](SSVP/ssvp_technical_walkthrough.md) |
| Install, train, troubleshoot | [`SSVP/ssvp_operations_guide.md`](SSVP/ssvp_operations_guide.md) |
| Compare the two trees | [`docs/STRUCTURE.md`](docs/STRUCTURE.md) |
| Merge / consolidate plan | [`docs/MERGE_PLAN.md`](docs/MERGE_PLAN.md) |
| Script-by-script inventory | [`docs/SCRIPT_INVENTORY.md`](docs/SCRIPT_INVENTORY.md) |

**Recommended canonical project:** `SSVP/ssvp/` — full submission bundle with datasets, baselines, distillation tooling, and top-level guides.

**Active dev fork:** `SSVPREFAC/DL_Project_refactor/` — caption evaluation (run22), cleaned script set, local venvs. Treat as a staging area until merged.

## Workspace layout

```
ssvpall/
├── README.md                 ← you are here
├── docs/                     ← monorepo guides (structure, merge plan, scripts)
├── SSVP/                     ← primary / submission-ready codebase
│   ├── ssvp/                 ← numbered submission folders (01–07)
│   ├── external/             ← AnomalyGPT, WinCLIP (top-level baselines)
│   ├── tools/                ← WinCLIP evaluation wrapper
│   ├── results/              ← WinCLIP benchmark outputs
│   ├── ssvp_technical_walkthrough.md
│   └── ssvp_operations_guide.md
└── SSVPREFAC/                ← refactor / experiment workspace
    ├── DL_Project_refactor/  ← same 01–07 layout as SSVP/ssvp
    ├── gpu_venv/             ← local GPU env (do not commit)
    ├── .venv/                ← local CPU env (do not commit)
    └── runs/                 ← ad-hoc experiment outputs
```

## What this project does

**SSVP** (*Synergistic Semantic-Visual Prompting*) fuses frozen **CLIP** (semantics) and **DINOv2** (structure) through three trainable modules:

- **HSVS** — cross-modal token fusion
- **VCPG** — vision-conditioned prompt generation
- **VTAM** — pixel anomaly maps and image-level scores

This repo extends the paper with:

1. Cable-focused MVTec AD protocol (70/20/10 resplit, anti-leakage validation)
2. Heavy-noise robustness evaluation
3. BLIP captioning with optional INT8 text-side compression
4. Ablation, compression-gating, and student distillation pipelines
5. WinCLIP / AnomalyCLIP / AnomalyGPT baselines

**Final recommended deployment:** checkpoint `run21_resplit_15es` + INT8 caption path via `live_demo_noisy_folder.py`. See [`SSVP/ssvp/05_results/RESULTS.md`](SSVP/ssvp/05_results/RESULTS.md).

## Environment setup

From the canonical tree:

```powershell
cd SSVP\ssvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r 03_code\requirements.txt
pip install bert-score>=0.3.13   # only for caption BERTScore eval (REFAC scripts)
python 03_code\scripts\test_shapes.py
```

For CUDA builds, install PyTorch first — see [`SSVP/ssvp_operations_guide.md`](SSVP/ssvp_operations_guide.md).

## Documentation map

| Document | Scope |
|----------|-------|
| [`docs/STRUCTURE.md`](docs/STRUCTURE.md) | Side-by-side tree comparison, what lives where |
| [`docs/MERGE_PLAN.md`](docs/MERGE_PLAN.md) | Phased consolidation roadmap |
| [`docs/SCRIPT_INVENTORY.md`](docs/SCRIPT_INVENTORY.md) | Every script in both trees |
| [`SSVP/ssvp/03_code/README.md`](SSVP/ssvp/03_code/README.md) | Runnable script reference |
| [`SSVP/ssvp/03_code/RUNALLEXPS.md`](SSVP/ssvp/03_code/RUNALLEXPS.md) | Experiment command cookbook |
