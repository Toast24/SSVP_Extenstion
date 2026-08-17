# Demo Tutorial: Run21 + LLM Compression From Scratch

This tutorial shows exactly how to run the main demo on a fresh machine:

- Train (or reuse) the run21-quality model.
- Run noisy-folder segmentation + captioning.
- Export visualizations, masks, and prompts/captions.

## 1) Hardware Requirements

### Minimum (works, slower)

- OS: Windows 10/11, Linux, or macOS.
- Python: 3.10+.
- RAM: 16 GB.
- GPU: CUDA-capable NVIDIA GPU with 8 GB VRAM.
- Disk: at least 25 GB free for data, checkpoints, and outputs.

### Recommended (for smoother training)

- RAM: 32 GB.
- GPU: 12 GB+ VRAM.
- Disk: 50 GB+.

Notes:

- Training in this repo is GPU-only.
- Demo inference can run on CPU but will be slower.

## 2) Clone and Environment Setup (PowerShell)

```powershell
git clone https://github.com/Toast24/DL_Project/
cd ssvp

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r 03_code/requirements.txt
```

## 3) Download MVTec AD (Official)

Download from:

- https://www.mvtec.com/company/research/datasets/mvtec-ad

After extraction, ensure cable category folders exist under 04_data/datasets/cable as:

- 04_data/datasets/cable/train/good/
- 04_data/datasets/cable/test/good/
- 04_data/datasets/cable/test/<defect_type>/
- 04_data/datasets/cable/ground_truth/<defect_type>/

## 4) Build the Resplit Dataset (70/20/10)

```powershell
python 03_code/scripts/prepare_cable_split.py --source 04_data/datasets/cable --output_root 04_data/datasets/cable_resplit --seed 42
```

Expected output:

- 04_data/datasets/cable_resplit/cable/
- 04_data/datasets/cable_resplit/cable/split_summary.txt

## 5) Produce Run21 Baseline Checkpoint

### Option A: Reproduce from scratch

```powershell
python 03_code/scripts/run_full_pipeline.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_dir 05_results/ablations/run21_resplit_15es --epochs 15 --num_vis_samples 5
```

This produces:

- 05_results/ablations/run21_resplit_15es/best_model.pth
- 05_results/ablations/run21_resplit_15es/eval_results/results.json
- 05_results/ablations/run21_resplit_15es/noise_eval/robustness_report.json
- 05_results/ablations/run21_resplit_15es/visualizations_with_captions/

### Option B: Reuse existing run21 checkpoint

If 05_results/ablations/run21_resplit_15es/best_model.pth already exists, you can skip training.

## 6) Run the Main Demo (Run21 + LLM Compression)

INT8 text compression for captioning is enabled by default in this demo script.

### Demo on a sample noisy folder

```powershell
python 03_code/scripts/live_demo_noisy_folder.py --input_folder 04_data/datasets/cable/test/combined --output_dir 05_results/ablations/live_demo --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --category cable --recursive
```

### Demo on your own noisy folder

```powershell
python 03_code/scripts/live_demo_noisy_folder.py --input_folder <PATH_TO_NOISY_IMAGES> --output_dir 05_results/ablations/live_demo_custom --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --category cable --recursive --max_images 0
```

Optional quality mode (slower):

```powershell
python 03_code/scripts/live_demo_noisy_folder.py --input_folder <PATH_TO_NOISY_IMAGES> --output_dir 05_results/ablations/live_demo_custom_prompt --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --category cable --recursive --caption_finetune
```

## 7) Understand Demo Outputs

Each processed image gets multiple artifacts under:

- 05_results/ablations/live_demo*/visualizations_with_captions/

Per-image files:

- *.viz.png: visualization panel with score and caption.
- *.heatmap.png: normalized anomaly heatmap.
- *.mask.png: binary mask from pixel threshold.
- *.overlay.png: red mask overlay on image.
- *.txt: generated caption/prompt text.

Run-level summary:

- 05_results/ablations/live_demo*/demo_summary.json

## 8) Quick Verification Commands

```powershell
Get-ChildItem 05_results/ablations/live_demo -Recurse -File | Select-Object -ExpandProperty FullName
Get-Content 05_results/ablations/live_demo/demo_summary.json
```

## 9) Common Issues

- Missing checkpoint:
  - Ensure 05_results/ablations/run21_resplit_15es/best_model.pth exists or pass --checkpoint explicitly.
- No images found:
  - Check --input_folder path and extensions.
- Slow captioning:
  - Keep INT8 compression enabled (default) and reduce --max_images.
- Training fails on CPU-only machine:
  - Use a CUDA-capable GPU system for training.

## 10) What To Read Next

1. [RUNALLEXPS.md](../03_code/RUNALLEXPS.md) for every experiment command.
2. [RESULTS.md](../05_results/RESULTS.md) for the full outcomes and analysis.
3. [README.md](../03_code/README.md) for project overview and script map.

