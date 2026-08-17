# SSVP Operations Guide: Datasets, Training, Validation & Testing

> This is the practical companion to the [SSVP Technical Walkthrough](file:///C:/Users/vivek/.gemini/antigravity/brain/777bba16-a2ad-4cde-9937-3fb527ef856c/ssvp_technical_walkthrough.md). It covers **how to use** the framework rather than how it works internally.

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Dataset Setup](#2-dataset-setup)
3. [Switching Datasets](#3-switching-datasets)
4. [Training Guide](#4-training-guide)
5. [Validation During Training](#5-validation-during-training)
6. [Testing & Evaluation](#6-testing--evaluation)
7. [Hardware Scaling Guide](#7-hardware-scaling-guide)
8. [Configuration Deep-Dive](#8-configuration-deep-dive)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites & Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3050 (4GB VRAM) | RTX 3060+ (8GB+ VRAM) |
| RAM | 16 GB | 32 GB |
| Storage | ~10 GB (MVTec-AD) | ~20 GB (multiple datasets) |
| Python | 3.10+ | 3.11 or 3.12 |

### Installation

```bash
cd ssvp

# Install PyTorch (with CUDA support for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install open-clip-torch timm scikit-learn scipy opencv-python matplotlib tqdm PyYAML
```

> [!NOTE]
> If you're on Python 3.13+, PyTorch may not have CUDA wheels for all platforms. Use the default `pip install torch torchvision` which will install the CPU or latest compatible CUDA version.

### Verify Installation

```bash
python test_shapes.py
```

Expected output: all 6 tests pass with 72,001,162 trainable parameters reported.

---

## 2. Dataset Setup

### 2.1 MVTec-AD (Default)

**Download**: [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (~4.9 GB)

**Setup**:
```bash
# Create data directory
mkdir -p data

# Extract the downloaded archive
# Expected result:
data/
└── mvtec_anomaly_detection/
    ├── bottle/
    │   ├── train/good/          # Normal training images
    │   ├── test/
    │   │   ├── good/            # Normal test images
    │   │   ├── broken_large/    # Defect type 1
    │   │   └── broken_small/    # Defect type 2
    │   └── ground_truth/
    │       ├── broken_large/    # Binary masks
    │       └── broken_small/
    ├── cable/
    ├── capsule/
    ... (15 categories total)
```

The 15 MVTec-AD categories are:

| Type | Categories |
|------|-----------|
| **Textures** (5) | carpet, grid, leather, tile, wood |
| **Objects** (10) | bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper |

**Verify dataset**:
```bash
python -c "
from data.mvtec import MVTecDataset
ds = MVTecDataset('./data/mvtec_anomaly_detection', split='test')
print(f'Test samples: {len(ds)}')
print(f'Sample keys: {list(ds[0].keys())}')
"
```

### 2.2 Statistics

| Split | Normal | Anomalous | Total | Has Masks |
|-------|--------|-----------|-------|-----------|
| Train | ~3,629 | 0 | ~3,629 | No |
| Test  | ~467 | ~1,258 | ~1,725 | Yes (for anomalous) |

---

## 3. Switching Datasets

### 3.1 Using a Different Data Root

The simplest change — point to a different location:

```bash
# Via command line
python train.py --config configs/default.yaml --data_root /path/to/mvtec_anomaly_detection

# Or modify configs/default.yaml:
data:
  data_root: "/path/to/mvtec_anomaly_detection"
```

### 3.2 Using Specific Categories

Train/test on a subset of categories:

```python
# In Python
from data.mvtec import get_mvtec_dataloaders, TEXTURE_CATEGORIES, OBJECT_CATEGORIES

# Only texture categories
train_loader, test_loader = get_mvtec_dataloaders(
    config,
    source_categories=list(TEXTURE_CATEGORIES),
    target_categories=list(TEXTURE_CATEGORIES),
)

# Only specific ones
train_loader, test_loader = get_mvtec_dataloaders(
    config,
    source_categories=["bottle", "cable", "capsule"],
    target_categories=["bottle", "cable", "capsule"],
)
```

Or via the inference script:
```bash
python inference.py --checkpoint outputs/best_model.pth --categories bottle cable capsule
```

### 3.3 Adding a New Dataset (e.g., VisA, BTAD, MPDD)

To add a completely new dataset, you need to create a new dataset class that follows the same interface. Here's a template:

**Step 1**: Create `data/custom_dataset.py`:

```python
"""
Custom Dataset Loader for SSVP.

Expected directory structure:
    custom_dataset/
    ├── category_1/
    │   ├── train/
    │   │   └── good/              # Normal training images
    │   ├── test/
    │   │   ├── good/              # Normal test images
    │   │   └── defect_type_1/     # Anomalous test images
    │   └── ground_truth/
    │       └── defect_type_1/     # Binary masks
    └── category_2/
        └── ...
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import DualResizeTransform, AugmentedTransform


CUSTOM_CATEGORIES = [
    "category_1", "category_2", "category_3",
    # ... list all your categories
]


class CustomDataset(Dataset):
    def __init__(self, data_root, categories=None, split="train",
                 img_size=518, mask_size=37, augment=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.categories = categories or CUSTOM_CATEGORIES

        if augment and split == "train":
            self.transform = AugmentedTransform(img_size, mask_size)
        else:
            self.transform = DualResizeTransform(img_size, mask_size)

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """
        Scan your directory structure and build a list of samples.
        Each sample must have:
            - image_path: str
            - mask_path: str or None
            - label: 0 (normal) or 1 (anomalous)
            - category: str
            - defect_type: str
        """
        for category in self.categories:
            cat_dir = os.path.join(self.data_root, category)
            if not os.path.isdir(cat_dir):
                continue

            if self.split == "train":
                # TODO: Adapt to your directory structure
                good_dir = os.path.join(cat_dir, "train", "good")
                for img_path in sorted(glob.glob(os.path.join(good_dir, "*.png"))):
                    self.samples.append({
                        "image_path": img_path,
                        "mask_path": None,
                        "label": 0,
                        "category": category,
                        "defect_type": "good",
                    })

            elif self.split == "test":
                # TODO: Adapt to your test directory structure
                test_dir = os.path.join(cat_dir, "test")
                gt_dir = os.path.join(cat_dir, "ground_truth")

                for defect_type in sorted(os.listdir(test_dir)):
                    defect_dir = os.path.join(test_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue

                    for img_path in sorted(glob.glob(os.path.join(defect_dir, "*.png"))):
                        img_name = os.path.splitext(os.path.basename(img_path))[0]

                        if defect_type == "good":
                            mask_path = None
                            label = 0
                        else:
                            # TODO: Adapt mask finding to your naming convention
                            mask_path = os.path.join(gt_dir, defect_type, img_name + ".png")
                            if not os.path.exists(mask_path):
                                mask_path = None
                            label = 1

                        self.samples.append({
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "label": label,
                            "category": category,
                            "defect_type": defect_type,
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = Image.open(sample["mask_path"]).convert("L")
        else:
            mask = None

        img_tensor, mask_tensor, mask_full = self.transform(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "mask_full": mask_full,
            "label": sample["label"],
            "category": sample["category"],
            "defect_type": sample["defect_type"],
            "image_path": sample["image_path"],
        }
```

**Step 2**: Register in `data/__init__.py`:

```python
from .mvtec import MVTecDataset
from .custom_dataset import CustomDataset
from .transforms import DualResizeTransform

__all__ = ["MVTecDataset", "CustomDataset", "DualResizeTransform"]
```

**Step 3**: Update config or modify `train.py` to use the new dataset:

```yaml
# configs/custom.yaml
data:
  dataset: "custom"
  data_root: "./data/custom_dataset"
  img_size: 518
  mask_size: 37
```

**Step 4**: Modify `train.py` to support the new dataset switch:

```python
# In train.py main(), replace get_mvtec_dataloaders with:
if config["data"]["dataset"] == "mvtec":
    from data.mvtec import get_mvtec_dataloaders
    train_loader, test_loader = get_mvtec_dataloaders(config)
elif config["data"]["dataset"] == "custom":
    from data.custom_dataset import CustomDataset
    train_dataset = CustomDataset(config["data"]["data_root"], split="train", augment=True)
    test_dataset = CustomDataset(config["data"]["data_root"], split="test")
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], ...)
    test_loader = DataLoader(test_dataset, batch_size=1, ...)
```

> [!IMPORTANT]
> **The critical contract**: Any custom dataset class must return a dict with at minimum these keys: `image` (Tensor), `mask` (Tensor), `mask_full` (Tensor), `label` (int), `category` (str). The training loop and evaluation pipeline depend on this exact interface.

### 3.4 Cross-Domain Zero-Shot Transfer

The paper's experimental protocol trains on one domain and evaluates on another:

| Protocol | Train On | Evaluate On |
|----------|----------|-------------|
| **MVTec-AD evaluation** | VisA dataset | MVTec-AD test set |
| **VisA evaluation** | MVTec-AD training set | VisA test set |

To implement this:
```python
# Train on MVTec, evaluate on custom/VisA
train_dataset = MVTecDataset("./data/mvtec_anomaly_detection", split="train", augment=True)
test_dataset = CustomDataset("./data/visa_dataset", split="test")
```

### 3.5 Resolution Considerations

If your custom dataset has images of different sizes, the transforms handle resizing automatically. However, be aware:

- All images are resized to `518x518` — very high-res images lose fine details, very small images get upscaled
- Masks must be binary (0 = normal, 1 = anomalous)
- Masks are auto-binarized at threshold 0.5 after resize
- Nearest-neighbor interpolation preserves binary mask boundaries

---

## 4. Training Guide

### 4.1 Basic Training

```bash
cd ssvp

# Train with default config (RTX 3050 optimized)
python train.py --config configs/default.yaml

# With custom data path
python train.py --config configs/default.yaml --data_root /path/to/mvtec

# With custom output directory
python train.py --config configs/default.yaml --output_dir ./my_experiment
```

### 4.2 Resume Training

```bash
# Resume from a checkpoint
python train.py --config configs/default.yaml --resume outputs/checkpoint_epoch10.pth
```

This restores:
- Model weights (including frozen backbone weights loaded from pretrained)
- Optimizer state (momentum, adaptive learning rates)
- Epoch counter (continues from the saved epoch + 1)
- Best AUROC (for checkpoint comparison)

### 4.3 What Happens During Training

```
Epoch 1/15 ━━━━━━━━━━━━━━━━━━━━ 100% | loss=12.34 seg=0.62 cls=0.89 alpha=0.00 lr=2.5e-04
Epoch 2/15 ━━━━━━━━━━━━━━━━━━━━ 100% | loss=8.21  seg=0.45 cls=0.71 alpha=0.12 lr=4.8e-04
Epoch 3/15 ━━━━━━━━━━━━━━━━━━━━ 100% | loss=5.67  seg=0.31 cls=0.52 alpha=0.34 lr=4.5e-04
  Validation — I-AUROC: 72.30% | F1-Max: 68.10% | AP: 71.80%
...
Epoch 15/15 ━━━━━━━━━━━━━━━━━━━ 100% | loss=2.14 seg=0.12 cls=0.23 alpha=0.81 lr=1.2e-05
  Validation — I-AUROC: 91.20% | F1-Max: 88.50% | AP: 90.10%
  * New best model saved!

Training complete! Best I-AUROC: 91.20%
```

**Key indicators to monitor**:

| Indicator | Good Sign | Bad Sign |
|-----------|-----------|----------|
| `loss` | Steadily decreasing | Stuck or oscillating wildly |
| `seg` | Decreasing below 0.3 | Stuck above 0.5 |
| `cls` | Decreasing below 0.3 | Stuck above 0.7 |
| `alpha` | Gradually increasing from 0 | Stays near 0 or explodes |
| `lr` | Follows cosine curve | N/A (automatic) |
| I-AUROC | Increasing each validation | Decreasing or flat |

### 4.4 Output Files

After training, the `outputs/` directory contains:

```
outputs/
├── best_model.pth           # Best checkpoint (by I-AUROC)
├── checkpoint_epoch5.pth    # Periodic checkpoint
├── checkpoint_epoch10.pth   # Periodic checkpoint
└── checkpoint_epoch15.pth   # Final checkpoint
```

Each checkpoint contains:
- `epoch`: int — which epoch it was saved at
- `model_state`: OrderedDict — full model state dict
- `optimizer_state`: dict — AdamW optimizer state
- `metrics`: dict — validation metrics at save time
- `best_auroc`: float — best I-AUROC seen so far

---

## 5. Validation During Training

### 5.1 When Validation Runs

By default, validation runs:
- Every **3 epochs** (epochs 3, 6, 9, 12, 15)
- On the **final epoch** (epoch 15)

### 5.2 What Validation Measures

During training, only **image-level metrics** are computed (faster than pixel-level):

| Metric | What it Tells You |
|--------|-------------------|
| **I-AUROC** | Overall discrimination ability between normal and anomalous |
| **F1-Max** | Best achievable F1 score at the optimal threshold |
| **AP** | Area under precision-recall curve |

### 5.3 Early Stopping (Manual)

The codebase doesn't implement automatic early stopping, but you can:

1. Watch I-AUROC during validation — if it plateaus for 6+ epochs, consider stopping
2. Use `Ctrl+C` to interrupt training — existing checkpoints are preserved
3. Resume with `--resume outputs/best_model.pth` if needed

### 5.4 Changing Validation Frequency

In [train.py](file:///c:/Users/vivek/Desktop/Antigrav/ssvp/train.py), modify line 269:

```python
# Validate every N epochs (change 3 to your preference)
if (epoch + 1) % 3 == 0 or epoch == train_cfg["epochs"] - 1:
```

---

## 6. Testing & Evaluation

### 6.1 Full Evaluation

```bash
# Basic evaluation
python inference.py --config configs/default.yaml --checkpoint outputs/best_model.pth

# With visualizations
python inference.py --config configs/default.yaml --checkpoint outputs/best_model.pth --visualize

# Evaluate specific categories only
python inference.py --checkpoint outputs/best_model.pth --categories bottle cable capsule

# Custom data path
python inference.py --checkpoint outputs/best_model.pth --data_root /path/to/mvtec
```

### 6.2 Understanding the Results Table

The evaluation script prints a comprehensive table:

```
========================================================================================
  SSVP Evaluation Results
========================================================================================

Category         I-AUROC   I-F1   I-AP |  P-AUROC  P-PRO   P-AP |     N
——————————————— ———————— ———————— ———— | ———————— ———————— ———— | —————
bottle              98.2   95.1   97.8 |     96.5    91.2   72.3 |    83
cable               87.3   82.4   85.9 |     94.1    85.7   51.2 |   150
...
——————————————— ———————— ———————— ———— | ———————— ———————— ———— | —————
AVERAGE             91.2   88.5   90.1 |     95.3    88.4   62.1 |  1725
========================================================================================
```

**Column explanations**:

| Column | Full Name | Range | Interpretation |
|--------|-----------|-------|----------------|
| I-AUROC | Image-level AUROC | 0-100% | Can the model tell normal from anomalous images? |
| I-F1 | Image-level F1-Max | 0-100% | Best precision-recall balance for classification |
| I-AP | Image-level Avg Precision | 0-100% | Ranking quality of anomaly scores |
| P-AUROC | Pixel-level AUROC | 0-100% | Can the model localize anomalous pixels? |
| P-PRO | Per-Region Overlap | 0-100% | Size-invariant localization quality |
| P-AP | Pixel-level Avg Precision | 0-100% | Pixel-level ranking quality |
| N | Sample count | - | Total test images for this category |

**Target performance** (from paper, with full resources): I-AUROC ~92-95%, P-AUROC ~96-97%

### 6.3 Results JSON

Results are also saved as structured JSON for programmatic analysis:

```json
{
  "per_category": {
    "bottle": {
      "image_level": {"auroc": 98.2, "f1_max": 95.1, "ap": 97.8},
      "pixel_level": {"auroc": 96.5, "pro": 91.2, "ap": 72.3},
      "n_samples": 83,
      "n_anomalous": 63
    },
    ...
  },
  "overall": {
    "image_level": {"auroc": 91.2, "f1_max": 88.5, "ap": 90.1},
    "pixel_level": {"auroc": 95.3, "pro": 88.4, "ap": 62.1},
    "n_samples": 1725
  }
}
```

### 6.4 Visualizing Anomaly Maps

With `--visualize`, the script saves heatmap overlays:

```
eval_results/
├── results.json
└── visualizations/
    ├── bottle/
    │   ├── sample_0.png      # 3-panel: input | heatmap | ground truth
    │   ├── sample_1.png
    │   └── ...
    ├── cable/
    └── ...
```

Each visualization shows:
1. **Left panel**: Original input image
2. **Center panel**: Anomaly heatmap overlay (jet colormap, alpha=0.5)
3. **Right panel**: Ground truth mask overlay (red)

### 6.5 Single Image Inference

For quick testing on individual images:

```python
import torch
from PIL import Image
from models.ssvp import SSVP
from data.transforms import DualResizeTransform
from utils import load_config, postprocess_anomaly_map, visualize_results

# Load model
config = load_config("configs/default.yaml")
model = SSVP(config).cuda()
checkpoint = torch.load("outputs/best_model.pth")
model.load_state_dict(checkpoint["model_state"], strict=False)
model.eval()

# Prepare image
transform = DualResizeTransform(img_size=518, mask_size=37)
image = Image.open("path/to/your/image.jpg").convert("RGB")
img_tensor, _, _ = transform(image)
img_tensor = img_tensor.unsqueeze(0).cuda()  # [1, 3, 518, 518]

# Inference
with torch.no_grad():
    outputs = model(img_tensor)

score = outputs["anomaly_score"].item()
anomaly_map = postprocess_anomaly_map(
    outputs["anomaly_map"], target_size=(518, 518), sigma=4.0
)

print(f"Anomaly score: {score:.4f}")
visualize_results(img_tensor[0].cpu(), anomaly_map[0], save_path="result.png")
```

---

## 7. Hardware Scaling Guide

### 7.1 RTX 3050 (4GB VRAM) — Default Configuration

```yaml
training:
  batch_size: 2
  grad_accum_steps: 4          # Effective: 8
  mixed_precision: true
  num_workers: 2
```

Expected: ~20 min/epoch, ~5 hours total for 15 epochs.

### 7.2 RTX 3060 (6-8GB VRAM)

```yaml
training:
  batch_size: 4
  grad_accum_steps: 2          # Effective: 8
  mixed_precision: true
  num_workers: 4
```

### 7.3 RTX 3080/4070 (10-12GB VRAM)

```yaml
training:
  batch_size: 8
  grad_accum_steps: 1          # No accumulation needed
  mixed_precision: true
  num_workers: 4
```

### 7.4 RTX 4090 / A100 (24+ GB VRAM)

```yaml
training:
  batch_size: 16
  grad_accum_steps: 1
  mixed_precision: true        # Can set to false if VRAM allows
  num_workers: 8
```

### 7.5 CPU Only (No GPU)

Training on CPU is technically possible but extremely slow (~10x slower). Disable mixed precision:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 8
  mixed_precision: false       # CPU doesn't support FP16 autocast
  num_workers: 2
```

### 7.6 Memory Optimization Tips

If you run out of GPU memory:

1. **Reduce batch_size to 1** and increase `grad_accum_steps` to 8
2. **Ensure mixed_precision is true** — halves activation memory
3. **Reduce num_workers to 0** — prevents memory pinning overhead
4. **Add `pin_memory: false`** to the config
5. **Monitor memory**: `nvidia-smi -l 1` in a separate terminal

---

## 8. Configuration Deep-Dive

### 8.1 Creating Custom Configs

Copy the default config and modify:

```bash
cp configs/default.yaml configs/my_experiment.yaml
```

Then use it:
```bash
python train.py --config configs/my_experiment.yaml
```

### 8.2 Key Parameters to Tune

**For better performance** (if you have more compute):

| Parameter | Default | Try |
|-----------|---------|-----|
| `training.epochs` | 15 | 20-30 |
| `training.batch_size` | 2 | 4-8 (with proportional accum reduction) |
| `vcpg.n_normal_prompts` | 3 | 5-8 (more prompt diversity) |
| `vcpg.n_abnormal_prompts` | 3 | 5-8 |
| `loss.lambda_reg` | 0.5 | 0.3-0.8 (controls prompt drift tolerance) |

**For faster experimentation** (lower quality, faster iteration):

| Parameter | Default | Try |
|-----------|---------|-----|
| `training.epochs` | 15 | 5-8 |
| `hsvs.num_layers` | 4 | 2-3 (fewer scales = faster) |
| `hsvs.num_heads` | 6 | 4 |
| `backbone.clip_feature_layers` | `[6,12,18,24]` | `[12,24]` (2 scales only) |
| `backbone.dino_feature_layers` | `[6,12,18,23]` | `[12,23]` (must match CLIP) |

> [!WARNING]
> If you change `num_layers` in HSVS, the corresponding backbone `feature_layers` lists must have the same length. The VTAM module also uses `num_layers`, so update all three consistently.

### 8.3 Loss Weight Tuning

The four loss components are balanced differently depending on your dataset:

```yaml
loss:
  focal_gamma: 2.0       # Increase to 3.0 if anomalies are very rare/small
  beta_kl: 0.1           # Decrease to 0.01 if VAE loss dominates early
  lambda_vae: 1.0        # Decrease if total loss is dominated by VAE term
  lambda_reg: 0.5        # Increase to 1.0 if prompts diverge too much
  xi_margin: 0.85        # Lower to 0.80 for more prompt flexibility
```

**Diagnostic**: If total loss is dominated by one component early in training, reduce its weight. The four losses should contribute roughly equally by mid-training.

### 8.4 Temperature Parameter

```yaml
vtam:
  tau: 0.07              # Anomaly probability sharpness
```

- `tau = 0.07` (default): Sharp probability distributions — high confidence
- `tau = 0.1-0.2`: Softer distributions — more uncertain predictions
- `tau < 0.05`: Very sharp — can cause numerical instability

The temperature is **learnable** (`nn.Parameter`), so it adapts during training. The initial value just sets the starting point.

---

## 9. Troubleshooting

### 9.1 CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of preference):
1. Reduce `batch_size` to 1  
2. Increase `grad_accum_steps` proportionally
3. Enable `mixed_precision: true`
4. Set `pin_memory: false` and `num_workers: 0`
5. Close other GPU applications (`nvidia-smi` to check)

### 9.2 DINOv2 Download Fails

**Symptom**: `torch.hub.load` fails with network errors

**Solutions**:
1. Check internet connectivity
2. Set `TORCH_HOME` to a writable directory: `export TORCH_HOME=/path/to/cache`
3. Pre-download the model manually:
   ```python
   import torch
   torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
   ```

### 9.3 Loss is NaN or Inf

**Symptom**: `loss=nan` during training

**Common causes and fixes**:
- **Learning rate too high**: Reduce `lr_prompt` and `lr_network` by 5-10x
- **Mixed precision underflow**: The `GradScaler` usually handles this, but try setting `mixed_precision: false` to diagnose
- **Empty dataset**: Check your data_root path is correct and has images
- **Temperature too low**: Ensure `vtam.tau` is >= 0.01

### 9.4 Training Loss Doesn't Decrease

**Symptom**: Loss stays flat or oscillates after several epochs

**Checklist**:
1. Verify data is loading correctly (check sample count matches expectations)
2. Check that both normal AND anomalous images are in the test set (training uses train split)
3. Verify learning rates: should start at 5e-4 / 1e-4, decay via cosine
4. Monitor `alpha`: if it stays at 0, the VCPG isn't injecting visual information
5. Try reducing weight decay from 0.01 to 0.001

### 9.5 Low Pixel-Level Metrics Despite Good Image-Level

**Symptom**: I-AUROC > 90% but P-AUROC < 80%

This indicates the model can detect anomalies but can't localize them. Possible fixes:
1. Increase `loss.focal_gamma` to 3.0 (more focus on hard pixels)
2. Decrease `vtam.gamma` to 0.3 (more emphasis on local evidence)
3. Increase `eval.gaussian_sigma` to 6.0-8.0 (smoother maps may align better with ground truth)
4. Check if your anomaly masks align correctly with the images

### 9.6 Unicode Errors on Windows

**Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Fix**: Run with UTF-8 mode:
```bash
python -X utf8 train.py --config configs/default.yaml
```

Or set the environment variable:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### 9.7 Slow Data Loading

**Symptom**: GPU utilization is low (< 50%), training is bottlenecked on data

**Fixes**:
1. Increase `num_workers` to 4-8 (match your CPU core count)
2. Enable `pin_memory: true`
3. Store dataset on an SSD, not HDD
4. On Windows, if `num_workers > 0` causes issues, try `num_workers: 0` (some Windows Python versions have multiprocessing issues)
