# SSVP Extension

**Robust zero-shot industrial anomaly detection, segmentation, and defect captioning using semantic-visual prompting.**

This project extends **SSVP (Synergistic Semantic-Visual Prompting)** for zero-shot anomaly detection on the **MVTec AD** benchmark, with additional experiments in noise robustness, defect captioning, model compression, and knowledge distillation.

The work focuses on moving beyond anomaly localization toward a practical inspection pipeline that can **detect defects, localize anomalous regions, and generate defect-oriented descriptions**.

---

## Overview

SSVP combines frozen visual and semantic representations from **DINOv2** and **CLIP**, with trainable modules for semantic-visual interaction and anomaly localization.

This extension adds:

- Noise robustness evaluation under heavy synthetic corruption
- Defect captioning for explainable inspection outputs
- INT8 dynamic quantization of the captioning text transformer
- Domain-specific prompt fine-tuning for defect descriptions
- Ablation studies on model components and robustness mechanisms
- Compression and knowledge distillation experiments
- Comparisons with anomaly detection baselines
- A deployment-oriented inference and demonstration pipeline

The primary experiments use a **cable-focused MVTec AD protocol** with a 70/20/10 split and leakage-aware validation.

---

## Key Results

The recommended `run21_resplit_15es` configuration achieves:

| Metric | Clean | Noisy |
|---|---:|---:|
| Image AUROC | **97.42** | **87.00** |
| Image F1-Max | **84.85** | **75.86** |
| Image AP | **93.62** | **81.33** |
| Pixel AUROC | **95.34** | **91.82** |
| Pixel PRO | 29.54 | 58.27 |
| Pixel AP | **50.89** | **34.91** |

Under heavy synthetic noise:

- Image AUROC decreases by **10.42 points**
- Image F1-Max decreases by **8.99 points**
- Image AP decreases by **12.28 points**
- Pixel AUROC decreases by **3.52 points**

The complete experiment ledger, ablations, and analysis are available in [`05_results/RESULTS.md`](SSVP/ssvp/05_results/RESULTS.md).

---

## Research Contributions

### 1. Robustness Evaluation

A standardized noisy evaluation protocol was introduced to measure performance degradation under heavy synthetic noise.

Both clean and noisy performance are reported to characterize robustness rather than evaluating only under ideal input conditions.

### 2. Defect Captioning

A captioning branch was added to provide natural-language descriptions alongside anomaly localization.

This extends the system from:

**anomaly detection → anomaly localization → defect description**

The captioning component provides an additional explainability layer for industrial inspection.

### 3. Caption-Side Model Compression

The captioning text transformer supports **INT8 dynamic quantization** of its linear layers.

The segmentation model remains unchanged while the captioning component receives a reduced compute footprint, providing a lighter deployment configuration.

### 4. Domain Prompt Fine-Tuning

Domain-specific prompts based on object category and defect type are used to improve caption specificity.

Example:

> close-up industrial inspection image of `<category>` showing `<defect>` defect

The caption fine-tuning pipeline freezes the majority of the BLIP model and updates the text decoder using token-level cross-entropy.

### 5. Compression and Distillation Experiments

The project evaluates pruning, compression gating, and student-distillation approaches using explicit acceptance criteria.

Experiments that did not meet the configured promotion criteria are retained and documented rather than being omitted.

### 6. Deployment-Oriented Inference

The final recommended configuration combines:

- The `run21_resplit_15es` segmentation checkpoint
- Noisy-image inference
- Defect caption generation
- Caption-side INT8 compression

This provides a single-checkpoint inference pipeline suitable for demonstration and further deployment experiments.

---

## Architecture

The system combines frozen representations from:

- **CLIP** for semantic information
- **DINOv2** for visual and structural information

with three trainable components:

### HSVS

**Cross-modal token fusion**

Combines semantic and visual representations to enable interaction between CLIP and DINOv2 features.

### VCPG

**Vision-conditioned prompt generation**

Uses visual information to generate task-relevant prompt representations.

### VTAM

**Visual Token Anomaly Mapping**

Produces pixel-level anomaly maps and image-level anomaly scores.

The resulting pipeline supports both anomaly localization and downstream defect captioning.

For a detailed architecture and implementation walkthrough, see [`ssvp_technical_walkthrough.md`](SSVP/ssvp_technical_walkthrough.md).

---

## Dataset and Evaluation

The primary benchmark is **MVTec AD**, using a cable-focused evaluation protocol.

The extended evaluation includes:

- Clean test evaluation
- Heavy synthetic noise evaluation
- Image-level anomaly detection
- Pixel-level anomaly localization
- Defect caption generation
- Ablation experiments
- Compression experiments
- Distillation experiments

### Evaluation Metrics

**Image-level**

- Image AUROC
- Image F1-Max
- Image AP

**Pixel-level**

- Pixel AUROC
- Pixel PRO
- Pixel AP

Caption experiments additionally evaluate qualitative caption quality and runtime characteristics.

---

## Repository Structure

    SSVP_Extension/
    │
    ├── docs/
    │   └── Project documentation
    │
    ├── SSVP/
    │   ├── external/
    │   │   └── External baseline implementations
    │   │
    │   ├── results/
    │   │   └── Benchmark outputs
    │   │
    │   ├── tools/
    │   │   └── Evaluation and utility tooling
    │   │
    │   ├── ssvp/
    │   │   ├── 01_admin/
    │   │   ├── 02_report/
    │   │   ├── 03_code/
    │   │   ├── 04_data/
    │   │   ├── 05_results/
    │   │   ├── 06_demo/
    │   │   └── 07_claims/
    │   │
    │   ├── Anomaly_SSVP_MLLM.pdf
    │   ├── paper_text.txt
    │   ├── ssvp_technical_walkthrough.md
    │   └── ssvp_operations_guide.md
    │
    ├── .gitignore
    └── README.md

### Project folders

| Directory | Purpose |
|---|---|
| `01_admin/` | Project and contribution information |
| `02_report/` | Research report |
| `03_code/` | Source code, configurations, and experiment scripts |
| `04_data/` | Dataset documentation and sample inputs |
| `05_results/` | Metrics, ablations, figures, and experiment artifacts |
| `06_demo/` | Demo instructions and example inputs |
| `07_claims/` | Reproduced vs. contributed work |

---

## Getting Started

### Environment

From the project directory:

    cd SSVP/ssvp
    python -m venv .venv

### Windows

    .\.venv\Scripts\Activate.ps1

### Linux / macOS

    source .venv/bin/activate

Install the required dependencies:

    pip install -r 03_code/requirements.txt

For caption evaluation:

    pip install bert-score>=0.3.13

Verify the installation:

    python 03_code/scripts/test_shapes.py

For CUDA/PyTorch setup, execution instructions, and troubleshooting, see [`ssvp_operations_guide.md`](SSVP/ssvp_operations_guide.md).

---

## Running the Demo

The recommended segmentation checkpoint is:

    05_results/ablations/run21_resplit_15es/best_model.pth

The primary demo entry point is:

    03_code/scripts/live_demo_noisy_folder.py

The recommended configuration uses caption-side INT8 compression.

For complete instructions, see [`06_demo/demo_instructions.md`](SSVP/ssvp/06_demo/demo_instructions.md).

---

## Results and Experiments

The complete experimental results are documented in [`05_results/RESULTS.md`](SSVP/ssvp/05_results/RESULTS.md).

This includes:

- Primary run results
- Clean vs. noisy evaluation
- Robustness analysis
- Captioning experiments
- Prompt fine-tuning experiments
- Compression experiments
- Compression-gating results
- Distillation experiments
- Ablation studies
- Final deployment recommendation

---

## Documentation

| Document | Description |
|---|---|
| [`03_code/README.md`](SSVP/ssvp/03_code/README.md) | Code and script reference |
| [`03_code/RUNALLEXPS.md`](SSVP/ssvp/03_code/RUNALLEXPS.md) | Experiment command reference |
| [`05_results/RESULTS.md`](SSVP/ssvp/05_results/RESULTS.md) | Results, ablations, and analysis |
| [`06_demo/demo_instructions.md`](SSVP/ssvp/06_demo/demo_instructions.md) | Demo instructions |
| [`ssvp_technical_walkthrough.md`](SSVP/ssvp_technical_walkthrough.md) | Architecture and implementation details |
| [`ssvp_operations_guide.md`](SSVP/ssvp_operations_guide.md) | Installation, execution, and troubleshooting |

---

## Reproducibility

The repository is organized to keep the experimental workflow reproducible:

1. Dataset and split information are documented in `04_data/`
2. Source code and experiment configurations are maintained in `03_code/`
3. Evaluation outputs and experiment artifacts are stored in `05_results/`
4. Demonstration instructions are provided in `06_demo/`
5. Project claims and attribution are documented in `07_claims/`

Individual experiment commands are available in [`03_code/RUNALLEXPS.md`](SSVP/ssvp/03_code/RUNALLEXPS.md).

---

## Research Status

**Status: Research / Experimental Implementation**

This repository contains the implementation, experimental pipeline, evaluation artifacts, ablation studies, and deployment-oriented demonstration developed as an extension of SSVP.

The reported results should be interpreted in the context of the documented **cable-focused MVTec AD evaluation protocol** and the corresponding clean/noisy experimental setup.

---

## Citation

If you use this implementation or build upon the extended experiments, please cite the original SSVP work.

The project paper and related material are available in [`Anomaly_SSVP_MLLM.pdf`](SSVP/Anomaly_SSVP_MLLM.pdf).

---

## Contact

**Kedar Athrey**

- [LinkedIn](https://www.linkedin.com/in/kedar-athrey-4324442ab)
- [GitHub](https://github.com/Toast24)
