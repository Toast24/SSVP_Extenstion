"""
SSVP Inference & Evaluation Script

Evaluates a trained SSVP model on the MVTec-AD test set.
Computes both image-level and pixel-level metrics, generates visualizations.

Usage:
    python 03_code/scripts/inference.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth
    python 03_code/scripts/inference.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/run21_resplit_15es/best_model.pth --visualize
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from path_utils import LOGS_DIR, default_config_path, ensure_import_paths

ensure_import_paths()

from models.ssvp import SSVP
from data.mvtec import MVTecDataset, MVTEC_CATEGORIES
from data.transforms import DualResizeTransform
from utils import (
    load_config, set_seed,
    compute_image_level_metrics, compute_pixel_level_metrics,
    postprocess_anomaly_map, visualize_results, denormalize_image,
)


@torch.no_grad()
def evaluate(model, test_loader, device, config, output_dir=None, visualize=False, calibration=None):
    """
    Full evaluation pipeline.

    Computes:
        - Image-level: AUROC, F1-Max, AP
        - Pixel-level: AUROC, PRO, AP

    Args:
        model: Trained SSVP model
        test_loader: Test dataloader
        device: torch device
        config: Config dict
        output_dir: Optional path for saving results/visualizations
        visualize: Whether to save visualization images

    Returns:
        results: dict with per-category and overall metrics
    """
    model.eval()
    eval_cfg = config.get("eval", {})
    sigma = eval_cfg.get("gaussian_sigma", 4.0)
    img_size = config["data"]["img_size"]
    tta_enabled = bool(eval_cfg.get("tta", {}).get("enabled", False))
    tta_hflip = bool(eval_cfg.get("tta", {}).get("hflip", True))
    clip_percentiles = tuple(eval_cfg.get("clip_percentiles", [1.0, 99.0])) if eval_cfg.get("clip_percentiles", None) else None
    median_ksize = int(eval_cfg.get("median_ksize", 0))

    # Accumulators — per-category
    category_data = {}

    for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
        images = batch["image"].to(device, non_blocking=True)
        masks_full = batch["mask_full"].numpy()  # [B, 1, H, W]
        labels = np.array(batch["label"])
        categories = batch["category"]
        image_paths = batch["image_path"]

        class_token_embedding = None
        if categories is not None:
            cat_name = categories[0] if isinstance(categories, (list, tuple)) else categories
            class_token_embedding = model.get_class_token_embedding(cat_name, device=device)

        # Forward pass (optionally with TTA)
        with autocast(enabled=config["training"].get("mixed_precision", True)):
            if tta_enabled:
                outputs_main = model(images, class_token_embedding=class_token_embedding)
                map_acc = outputs_main["anomaly_map"]
                score_acc = outputs_main["anomaly_score"]
                n_views = 1

                if tta_hflip:
                    images_flip = torch.flip(images, dims=[3])
                    outputs_flip = model(images_flip, class_token_embedding=class_token_embedding)
                    map_flip = torch.flip(outputs_flip["anomaly_map"], dims=[3])
                    map_acc = map_acc + map_flip
                    score_acc = score_acc + outputs_flip["anomaly_score"]
                    n_views += 1

                outputs = {
                    "anomaly_map": map_acc / n_views,
                    "anomaly_score": score_acc / n_views,
                }
            else:
                outputs = model(images, class_token_embedding=class_token_embedding)

        # Extract predictions
        anomaly_scores = outputs["anomaly_score"].cpu().numpy()  # [B]
        anomaly_maps = torch.sigmoid(outputs["anomaly_map"])  # [B, 1, H', W'] probabilities

        # Post-process anomaly maps
        processed_maps = postprocess_anomaly_map(
            anomaly_maps,
            target_size=(img_size, img_size),
            sigma=sigma,
            clip_percentiles=clip_percentiles,
            median_ksize=median_ksize,
        )  # [B, H, W]

        # Store per-category
        for i in range(len(labels)):
            cat = categories[i]
            if cat not in category_data:
                category_data[cat] = {
                    "scores": [], "labels": [],
                    "maps": [], "masks": [],
                    "paths": [],
                }

            category_data[cat]["scores"].append(anomaly_scores[i])
            category_data[cat]["labels"].append(labels[i])
            category_data[cat]["maps"].append(processed_maps[i])
            category_data[cat]["masks"].append(masks_full[i, 0])  # [H, W]
            category_data[cat]["paths"].append(image_paths[i])

        # Visualize (first 5 anomalous samples per category)
        if visualize and output_dir:
            for i in range(len(labels)):
                if labels[i] == 1:
                    cat = categories[i]
                    vis_dir = os.path.join(output_dir, "visualizations", cat)
                    existing = len([f for f in os.listdir(vis_dir)] if os.path.isdir(vis_dir) else [])

                    if existing < 5:
                        img_np = denormalize_image(images[i].cpu())
                        vis_path = os.path.join(vis_dir, f"sample_{existing}.png")
                        visualize_results(
                            img_np, processed_maps[i],
                            mask=masks_full[i, 0],
                            save_path=vis_path,
                            title=f"{cat} — Score: {anomaly_scores[i]:.3f}"
                        )

    # ── Compute metrics ──
    results = {"per_category": {}, "overall": {}}
    all_scores, all_labels = [], []
    all_maps, all_masks = [], []

    for cat in sorted(category_data.keys()):
        data = category_data[cat]
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])
        maps = np.stack(data["maps"])
        masks = np.stack(data["masks"])

        # Image-level metrics
        img_metrics = compute_image_level_metrics(scores, labels)

        # Pixel-level metrics
        pix_metrics = compute_pixel_level_metrics(maps, masks)

        results["per_category"][cat] = {
            "image_level": img_metrics,
            "pixel_level": pix_metrics,
            "n_samples": len(labels),
            "n_anomalous": int(labels.sum()),
        }

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        all_maps.append(maps)
        all_masks.append(masks)

    # Overall metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_maps = np.concatenate(all_maps, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    results["overall"]["image_level"] = compute_image_level_metrics(all_scores, all_labels)
    results["overall"]["pixel_level"] = compute_pixel_level_metrics(all_maps, all_masks)
    results["overall"]["n_samples"] = len(all_labels)

    if calibration is not None:
        image_thr = float(calibration.get("image_threshold", 0.5))
        pixel_thr = float(calibration.get("pixel_threshold", 0.5))

        pred_labels = (all_scores >= image_thr).astype(int)
        pred_pixels = (all_maps >= pixel_thr).astype(int)

        results["overall"]["thresholded"] = {
            "image_threshold": image_thr,
            "pixel_threshold": pixel_thr,
            "image_accuracy": float(accuracy_score(all_labels, pred_labels) * 100.0),
            "image_precision": float(precision_score(all_labels, pred_labels, zero_division=0) * 100.0),
            "image_recall": float(recall_score(all_labels, pred_labels, zero_division=0) * 100.0),
            "image_f1": float(f1_score(all_labels, pred_labels, zero_division=0) * 100.0),
            "pixel_accuracy": float((pred_pixels.flatten() == all_masks.flatten().astype(int)).mean() * 100.0),
        }

    return results


def print_results(results):
    """Pretty-print evaluation results in table format."""
    print(f"\n{'='*90}")
    print(f"  SSVP Evaluation Results")
    print(f"{'='*90}")

    # Per-category table
    print(f"\n{'Category':<15} {'I-AUROC':>8} {'I-F1':>8} {'I-AP':>8} │ "
          f"{'P-AUROC':>8} {'P-PRO':>8} {'P-AP':>8} │ {'N':>5}")
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} │ {'─'*8} {'─'*8} {'─'*8} │ {'─'*5}")

    for cat in sorted(results["per_category"].keys()):
        cat_data = results["per_category"][cat]
        img = cat_data["image_level"]
        pix = cat_data["pixel_level"]
        n = cat_data["n_samples"]

        print(f"{cat:<15} "
              f"{img['auroc']:>8.1f} {img['f1_max']:>8.1f} {img['ap']:>8.1f} │ "
              f"{pix['auroc']:>8.1f} {pix['pro']:>8.1f} {pix['ap']:>8.1f} │ "
              f"{n:>5}")

    # Overall
    overall = results["overall"]
    img = overall["image_level"]
    pix = overall["pixel_level"]
    n = overall["n_samples"]

    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} │ {'─'*8} {'─'*8} {'─'*8} │ {'─'*5}")
    print(f"{'AVERAGE':<15} "
          f"{img['auroc']:>8.1f} {img['f1_max']:>8.1f} {img['ap']:>8.1f} │ "
          f"{pix['auroc']:>8.1f} {pix['pro']:>8.1f} {pix['ap']:>8.1f} │ "
          f"{n:>5}")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="SSVP Evaluation")
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(LOGS_DIR / "inference_eval"))
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Specific categories to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_file", type=str, default=None,
                        help="Optional calibration thresholds JSON from validation split")
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading SSVP model...")
    model = SSVP(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Test dataset
    categories = args.categories or MVTEC_CATEGORIES
    test_dataset = MVTecDataset(
        data_root=config["data"]["data_root"],
        categories=categories,
        split="test",
        img_size=config["data"]["img_size"],
        mask_size=config["data"]["mask_size"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    calibration = None
    if args.calibration_file and os.path.exists(args.calibration_file):
        with open(args.calibration_file, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        print(f"Loaded calibration thresholds from: {args.calibration_file}")

    # Evaluate
    results = evaluate(
        model, test_loader, device, config,
        output_dir=args.output_dir,
        visualize=args.visualize,
        calibration=calibration,
    )

    # Print and save
    print_results(results)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
