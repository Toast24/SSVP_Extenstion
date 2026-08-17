"""
Shared caption-target helpers, ported from SSVPREFAC/DL_Project_refactor/03_code/scripts/run_full_pipeline.py
(see docs/MERGE_PLAN.md Phase 2/3.1). Extracted here so evaluate_captions.py / caption_compare.py
don't need to import from run_full_pipeline.py directly.
"""

import torch
import numpy as np


def _normalize_token_label(value):
    return str(value or "").strip().lower().replace("_", " ")


def _build_domain_caption(category, defect_type):
    category_label = _normalize_token_label(category) or "object"
    defect_label = _normalize_token_label(defect_type) or "good"

    if defect_label == "good":
        return f"photo of {category_label} with no visible defect"
    return f"photo of {category_label} with a visible defect"


def _mask_to_spatial_location(mask_full):
    if mask_full is None:
        return "center"

    if torch.is_tensor(mask_full):
        mask_np = mask_full.detach().cpu().numpy()
    else:
        mask_np = mask_full

    mask_np = (mask_np > 0).astype("uint8")
    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return "center"

    x_center = float((xs.min() + xs.max()) * 0.5 / max(mask_np.shape[1] - 1, 1))
    y_center = float((ys.min() + ys.max()) * 0.5 / max(mask_np.shape[0] - 1, 1))

    if y_center < 0.33:
        vertical = "upper"
    elif y_center > 0.66:
        vertical = "lower"
    else:
        vertical = "middle"

    if x_center < 0.33:
        horizontal = "left"
    elif x_center > 0.66:
        horizontal = "right"
    else:
        horizontal = "center"

    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical} {horizontal}"


def _build_spatial_defect_caption(category, defect_type, mask_full=None):
    defect_label = _normalize_token_label(defect_type) or "good"
    if defect_label == "good":
        return "no anomaly"

    location = _mask_to_spatial_location(mask_full)
    return f"anomaly at {location}"


def _build_caption_target(sample, target_mode="spatial"):
    if str(target_mode).strip().lower() == "spatial":
        return _build_spatial_defect_caption(
            sample.get("category"),
            sample.get("defect_type"),
            sample.get("mask_full"),
        )
    return _build_domain_caption(sample.get("category"), sample.get("defect_type"))
