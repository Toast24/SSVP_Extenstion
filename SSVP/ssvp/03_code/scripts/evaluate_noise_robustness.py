"""
Evaluate checkpoint robustness on heavily noised test images and compare to clean results.

Usage:
    python 03_code/scripts/evaluate_noise_robustness.py --config 03_code/configs/default.yaml --checkpoint 05_results/ablations/runX/best_model.pth \
            --output_dir 05_results/ablations/runX/noise_eval --data_root 04_data/datasets/cable_resplit --categories cable \
            --clean_results 05_results/ablations/runX/eval_results/results.json --calibration_file 05_results/ablations/runX/calibration_thresholds.json
"""

import argparse
import io
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast
from tqdm import tqdm

from path_utils import default_config_path, ensure_import_paths

ensure_import_paths()

from data.mvtec import MVTEC_CATEGORIES, MVTecDataset
from models.ssvp import SSVP
from utils import (
    compute_image_level_metrics,
    compute_pixel_level_metrics,
    denormalize_image,
    load_config,
    postprocess_anomaly_map,
    set_seed,
    visualize_results,
)


def _jpeg_degrade(img_uint8, quality=20):
    pil = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _to_float01(img_uint8):
    return img_uint8.astype(np.float32) / 255.0


def _to_uint8(img_float):
    return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)


def _add_heavy_noise(img_uint8, gaussian_std=0.25, salt_pepper_prob=0.03, jpeg_quality=20, blur_sigma=2.0):
    from scipy.ndimage import gaussian_filter

    x = _to_float01(img_uint8)

    # Gaussian noise
    x = np.clip(x + np.random.normal(0.0, gaussian_std, size=x.shape).astype(np.float32), 0.0, 1.0)

    # Salt-and-pepper noise
    if salt_pepper_prob > 0:
        rnd = np.random.rand(*x.shape[:2])
        x[rnd < (salt_pepper_prob / 2)] = 0.0
        x[rnd > (1.0 - salt_pepper_prob / 2)] = 1.0

    # Blur
    if blur_sigma > 0:
        for c in range(3):
            x[:, :, c] = gaussian_filter(x[:, :, c], sigma=blur_sigma)

    # JPEG artifacts
    if jpeg_quality is not None and jpeg_quality > 0:
        x = _to_float01(_jpeg_degrade(_to_uint8(x), quality=jpeg_quality))

    return _to_uint8(x)


def _normalize_like_pipeline(img_uint8):
    x = _to_float01(img_uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


@torch.no_grad()
def evaluate_noisy(model, dataset, device, config, out_dir, calibration=None):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    eval_cfg = config.get("eval", {})
    rob_cfg = config.get("robustness", {}).get("noise_eval", {})
    sigma = eval_cfg.get("gaussian_sigma", 1.5)
    img_size = config["data"]["img_size"]
    clip_percentiles = tuple(eval_cfg.get("clip_percentiles", [1.0, 99.0])) if eval_cfg.get("clip_percentiles", None) else None
    median_ksize = int(eval_cfg.get("median_ksize", 0))
    num_vis = int(rob_cfg.get("num_visualizations", 5))

    all_scores = []
    all_labels = []
    all_maps = []
    all_masks = []

    vis_saved = 0

    for idx in tqdm(range(len(dataset)), desc="Noisy eval", ncols=100):
        item = dataset[idx]
        img_clean = denormalize_image(item["image"])  # uint8 HWC

        noisy_uint8 = _add_heavy_noise(
            img_clean,
            gaussian_std=float(rob_cfg.get("gaussian_std", 0.25)),
            salt_pepper_prob=float(rob_cfg.get("salt_pepper_prob", 0.03)),
            jpeg_quality=int(rob_cfg.get("jpeg_quality", 20)),
            blur_sigma=float(rob_cfg.get("blur_sigma", 2.0)),
        )

        noisy_tensor = _normalize_like_pipeline(noisy_uint8).unsqueeze(0).to(device)
        class_emb = model.get_class_token_embedding(item.get("category", None), device=device)

        with autocast(enabled=config["training"].get("mixed_precision", True)):
            outputs = model(noisy_tensor, class_token_embedding=class_emb)

        score = float(outputs["anomaly_score"].detach().cpu().flatten()[0].item())
        proc_map = postprocess_anomaly_map(
            torch.sigmoid(outputs["anomaly_map"]),
            target_size=(img_size, img_size),
            sigma=sigma,
            clip_percentiles=clip_percentiles,
            median_ksize=median_ksize,
        )[0]

        all_scores.append(score)
        all_labels.append(int(item["label"]))
        all_maps.append(proc_map)
        all_masks.append(item["mask_full"][0].cpu().numpy())

        if vis_saved < num_vis and int(item["label"]) == 1:
            save_path = os.path.join(vis_dir, f"sample_{vis_saved}.png")
            visualize_results(
                noisy_uint8,
                proc_map,
                mask=item["mask_full"][0].cpu().numpy(),
                save_path=save_path,
                title=f"Noisy | Score: {score:.3f}",
            )
            vis_saved += 1

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_maps = np.stack(all_maps)
    all_masks = np.stack(all_masks)

    results = {
        "overall": {
            "image_level": compute_image_level_metrics(all_scores, all_labels),
            "pixel_level": compute_pixel_level_metrics(all_maps, all_masks),
            "n_samples": int(len(all_labels)),
        }
    }

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

    with open(os.path.join(out_dir, "noise_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def build_robustness_report(clean_results, noisy_results):
    c = clean_results["overall"]
    n = noisy_results["overall"]
    report = {
        "clean": c,
        "noisy": n,
        "delta": {
            "image_auroc": n["image_level"]["auroc"] - c["image_level"]["auroc"],
            "image_f1_max": n["image_level"]["f1_max"] - c["image_level"]["f1_max"],
            "image_ap": n["image_level"]["ap"] - c["image_level"]["ap"],
            "pixel_auroc": n["pixel_level"]["auroc"] - c["pixel_level"]["auroc"],
            "pixel_pro": n["pixel_level"]["pro"] - c["pixel_level"]["pro"],
            "pixel_ap": n["pixel_level"]["ap"] - c["pixel_level"]["ap"],
        },
    }

    if "thresholded" in c and "thresholded" in n:
        report["delta"].update({
            "image_accuracy": n["thresholded"]["image_accuracy"] - c["thresholded"]["image_accuracy"],
            "image_f1": n["thresholded"]["image_f1"] - c["thresholded"]["image_f1"],
            "pixel_accuracy": n["thresholded"]["pixel_accuracy"] - c["thresholded"]["pixel_accuracy"],
        })

    return report


def main():
    parser = argparse.ArgumentParser(description="Noisy robustness evaluation")
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--clean_results", type=str, default=None)
    parser.add_argument("--calibration_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSVP(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    categories = args.categories or MVTEC_CATEGORIES
    dataset = MVTecDataset(
        data_root=config["data"]["data_root"],
        categories=categories,
        split="test",
        img_size=config["data"]["img_size"],
        mask_size=config["data"]["mask_size"],
    )

    calibration = None
    if args.calibration_file and os.path.exists(args.calibration_file):
        with open(args.calibration_file, "r", encoding="utf-8") as f:
            calibration = json.load(f)

    noisy_results = evaluate_noisy(model, dataset, device, config, args.output_dir, calibration=calibration)

    if args.clean_results and os.path.exists(args.clean_results):
        with open(args.clean_results, "r", encoding="utf-8") as f:
            clean_results = json.load(f)
        report = build_robustness_report(clean_results, noisy_results)
        with open(os.path.join(args.output_dir, "robustness_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Robustness report saved to: {os.path.join(args.output_dir, 'robustness_report.json')}")

    print(f"Noisy results saved to: {os.path.join(args.output_dir, 'noise_results.json')}")


if __name__ == "__main__":
    main()
