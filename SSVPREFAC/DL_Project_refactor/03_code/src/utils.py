"""
Utility functions for SSVP — Metrics, Visualization, and Helpers.

Evaluation metrics (§4.1):
    Image-level: AUROC, F1-Max, AP
    Pixel-level:  AUROC, PRO, AP
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")  # Non-interactive backend


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_image_level_metrics(scores, labels):
    """
    Compute image-level anomaly detection metrics.

    Args:
        scores: np.array [N] — anomaly scores
        labels: np.array [N] — binary labels (0=normal, 1=anomalous)

    Returns:
        dict with AUROC, F1-Max, AP
    """
    if len(np.unique(labels)) < 2:
        return {"auroc": 0.0, "f1_max": 0.0, "ap": 0.0}

    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    # F1-Max: find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    f1_max = np.max(f1_scores)

    return {"auroc": auroc * 100, "f1_max": f1_max * 100, "ap": ap * 100}


def compute_pixel_level_metrics(anomaly_maps, masks):
    """
    Compute pixel-level anomaly detection metrics.

    Args:
        anomaly_maps: np.array [N, H, W] — predicted anomaly maps
        masks:        np.array [N, H, W] — ground truth binary masks

    Returns:
        dict with AUROC, PRO, AP
    """
    # Flatten for pixel-wise metrics
    preds_flat = anomaly_maps.flatten()
    masks_flat = masks.flatten().astype(int)

    if len(np.unique(masks_flat)) < 2:
        return {"auroc": 0.0, "pro": 0.0, "ap": 0.0}

    auroc = roc_auc_score(masks_flat, preds_flat)
    ap = average_precision_score(masks_flat, preds_flat)

    # PRO: Per-Region Overlap
    pro = compute_pro_score(anomaly_maps, masks)

    return {"auroc": auroc * 100, "pro": pro * 100, "ap": ap * 100}


def compute_pro_score(anomaly_maps, masks, num_thresholds=200):
    """
    Compute Per-Region Overlap (PRO) score.

    PRO treats anomalies of varying sizes with equal importance by computing
    the mean overlap ratio per connected component at each threshold.

    Args:
        anomaly_maps: np.array [N, H, W]
        masks:        np.array [N, H, W]
        num_thresholds: Number of thresholds for integration

    Returns:
        pro_score: float ∈ [0, 1]
    """
    from scipy.ndimage import label as ndimage_label

    # Normalize anomaly maps to [0, 1]
    flat = anomaly_maps.flatten()
    min_val, max_val = flat.min(), flat.max()
    if max_val - min_val < 1e-8:
        return 0.0
    norm_maps = (anomaly_maps - min_val) / (max_val - min_val)

    # Generate thresholds
    thresholds = np.linspace(0, 1, num_thresholds + 1)[1:]

    pro_values = []
    fpr_values = []

    for thresh in thresholds:
        binary_pred = (norm_maps >= thresh).astype(int)

        # Compute per-region overlap
        region_overlaps = []
        for i in range(len(masks)):
            if masks[i].max() == 0:
                continue

            # Label connected components in ground truth
            labeled_mask, num_regions = ndimage_label(masks[i])

            for region_id in range(1, num_regions + 1):
                region = (labeled_mask == region_id)
                region_size = region.sum()
                if region_size == 0:
                    continue
                overlap = (binary_pred[i] * region).sum() / region_size
                region_overlaps.append(overlap)

        if len(region_overlaps) > 0:
            mean_overlap = np.mean(region_overlaps)
        else:
            mean_overlap = 0.0

        # FPR for this threshold
        normal_pixels = (masks == 0)
        if normal_pixels.sum() > 0:
            fpr = binary_pred[normal_pixels].sum() / normal_pixels.sum()
        else:
            fpr = 0.0

        pro_values.append(mean_overlap)
        fpr_values.append(fpr)

    # Integrate PRO vs FPR curve (up to FPR=0.3)
    pro_values = np.array(pro_values)
    fpr_values = np.array(fpr_values)

    # Sort by FPR
    sort_idx = np.argsort(fpr_values)
    fpr_sorted = fpr_values[sort_idx]
    pro_sorted = pro_values[sort_idx]

    # Clip at FPR=0.3
    valid = fpr_sorted <= 0.3
    if valid.sum() < 2:
        return 0.0

    fpr_valid = fpr_sorted[valid]
    pro_valid = pro_sorted[valid]

    # Trapezoidal integration (NumPy>=2.0 prefers trapezoid)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(pro_valid, fpr_valid)
    else:
        area = np.trapz(pro_valid, fpr_valid)
    pro_score = area / 0.3

    return pro_score


# ═══════════════════════════════════════════════════════════════════════════
#  Post-Processing
# ═══════════════════════════════════════════════════════════════════════════

def postprocess_anomaly_map(
    anomaly_map,
    target_size,
    sigma=4.0,
    clip_percentiles=None,
    median_ksize=0,
):
    """
    Post-process anomaly map: upsample + Gaussian smoothing.

    Args:
        anomaly_map: torch.Tensor [B, 1, H, W] or [1, H, W]
        target_size: tuple (H, W) for upsampling
        sigma: Gaussian blur sigma
        clip_percentiles: Optional (low, high) tuple for robust clipping
        median_ksize: Optional median filter kernel size (>1 enables)

    Returns:
        smoothed_map: np.array [B, H, W] or [H, W]
    """
    if anomaly_map.dim() == 3:
        anomaly_map = anomaly_map.unsqueeze(0)

    # Upsample
    upsampled = F.interpolate(
        anomaly_map, size=target_size,
        mode="bilinear", align_corners=False
    )

    # Convert to numpy
    maps_np = upsampled.squeeze(1).cpu().numpy()  # [B, H, W]

    # Apply robust clipping + smoothing
    smoothed = np.zeros_like(maps_np)
    for i in range(maps_np.shape[0]):
        m = maps_np[i]
        if clip_percentiles is not None:
            lo, hi = clip_percentiles
            lo_v, hi_v = np.percentile(m, [lo, hi])
            m = np.clip(m, lo_v, hi_v)
            denom = max(hi_v - lo_v, 1e-8)
            m = (m - lo_v) / denom

        m = gaussian_filter(m, sigma=sigma)
        if median_ksize and int(median_ksize) > 1:
            m = median_filter(m, size=int(median_ksize))

        smoothed[i] = m

    return smoothed


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_results(image, anomaly_map, mask=None, save_path=None, title=None):
    """
    Visualize input image, anomaly heatmap, and optional ground truth.

    Args:
        image:       np.array [H, W, 3] (RGB, 0-255) or torch.Tensor [3, H, W]
        anomaly_map: np.array [H, W] — anomaly probability
        mask:        np.array [H, W] — ground truth mask (optional)
        save_path:   Path to save visualization
        title:       Title string
    """
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)

    n_cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Anomaly heatmap
    axes[1].imshow(image)
    heatmap = axes[1].imshow(anomaly_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    # Ground truth
    if mask is not None:
        axes[2].imshow(image)
        axes[2].imshow(mask, cmap="Reds", alpha=0.5)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize a torch image tensor to numpy RGB (0-255).

    Args:
        tensor: [3, H, W] — normalized image tensor
    Returns:
        image: [H, W, 3] — uint8 numpy array
    """
    img = tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Config Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path):
    """Load YAML configuration file."""
    import yaml
    # Try UTF-8 first, fall back to system encoding if needed
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, "r", encoding="cp1252") as f:
            config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
