"""
SSVP Training Script — Algorithm 3

Training protocol:
    - Cross-Domain Zero-Shot Transfer (§4.2)
    - Gradient accumulation for RTX 3050 (batch_size=2, accum=4 → effective=8)
    - Mixed precision (FP16) for memory savings
    - Dual learning rates: prompts (5e-4) vs network (1e-4)
    - Cosine annealing scheduler with warmup

Usage:
    python 03_code/scripts/train.py --config 03_code/configs/default.yaml
    python 03_code/scripts/train.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit
"""

import os
import sys
import time
import argparse
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

from path_utils import LOGS_DIR, default_config_path, ensure_import_paths

ensure_import_paths()

from models.ssvp import SSVP
from models.losses import SSVPLoss
from data.mvtec import get_mvtec_train_val_dataloaders
from utils import load_config, set_seed, postprocess_anomaly_map, visualize_results, denormalize_image


def _perturb_for_consistency(images, noise_std=0.04):
    if noise_std <= 0:
        return images
    noisy = images + torch.randn_like(images) * noise_std
    return torch.clamp(noisy, -5.0, 5.0)


def _perturb_for_noisy_calibration(images, eval_cfg, robustness_cfg):
    calib_cfg = eval_cfg.get("calibration", {})
    if not bool(calib_cfg.get("use_noisy_val", True)):
        return images

    noise_eval_cfg = robustness_cfg.get("noise_eval", {})
    noise_std = float(calib_cfg.get("noise_std", noise_eval_cfg.get("gaussian_std", 0.15)))
    salt_pepper_prob = float(calib_cfg.get("salt_pepper_prob", noise_eval_cfg.get("salt_pepper_prob", 0.01)))

    noisy = images + torch.randn_like(images) * noise_std

    if salt_pepper_prob > 0:
        rnd = torch.rand_like(noisy[:, :1, :, :])
        salt = rnd > (1.0 - salt_pepper_prob / 2.0)
        pepper = rnd < (salt_pepper_prob / 2.0)
        noisy = torch.where(salt.expand_as(noisy), torch.full_like(noisy, 5.0), noisy)
        noisy = torch.where(pepper.expand_as(noisy), torch.full_like(noisy, -5.0), noisy)

    return torch.clamp(noisy, -5.0, 5.0)


def _binary_f1_from_threshold(scores, labels, threshold):
    pred = (scores >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _robust_threshold(clean_values, noisy_values, labels, num_candidates=256, max_samples=None, seed=42):
    clean_values = np.asarray(clean_values)
    noisy_values = np.asarray(noisy_values)
    labels = np.asarray(labels).astype(np.int32)

    if clean_values.size == 0:
        return 0.5, 0.0, 0.0

    if max_samples is not None and clean_values.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(clean_values.size, size=max_samples, replace=False)
        clean_values = clean_values[idx]
        noisy_values = noisy_values[idx]
        labels = labels[idx]

    lo = float(min(clean_values.min(), noisy_values.min()))
    hi = float(max(clean_values.max(), noisy_values.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5, 0.0, 0.0

    candidates = np.linspace(lo, hi, int(max(num_candidates, 16)))

    best_thr = float(candidates[0])
    best_obj = -1.0
    best_clean_f1 = 0.0
    best_noisy_f1 = 0.0

    for thr in candidates:
        f1_clean = _binary_f1_from_threshold(clean_values, labels, thr)
        f1_noisy = _binary_f1_from_threshold(noisy_values, labels, thr)
        obj = min(f1_clean, f1_noisy) + 0.25 * (f1_clean + f1_noisy)
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
            best_clean_f1 = float(f1_clean)
            best_noisy_f1 = float(f1_noisy)

    return best_thr, best_clean_f1, best_noisy_f1


@torch.no_grad()
def derive_validation_thresholds(model, val_loader, device, config):
    """Derive image/pixel thresholds from validation split by maximizing F1."""
    model.eval()

    eval_cfg = config.get("eval", {})
    robustness_cfg = config.get("robustness", {})
    calib_cfg = eval_cfg.get("calibration", {})
    use_noisy_val = bool(calib_cfg.get("use_noisy_val", False))
    sigma = eval_cfg.get("gaussian_sigma", 1.5)
    img_size = config["data"]["img_size"]
    clip_percentiles = tuple(eval_cfg.get("clip_percentiles", [1.0, 99.0])) if eval_cfg.get("clip_percentiles", None) else None
    median_ksize = int(eval_cfg.get("median_ksize", 0))

    all_scores = []
    all_scores_noisy = []
    all_labels = []
    all_maps = []
    all_maps_noisy = []
    all_masks = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.float32, device=device)
        masks_full = batch["mask_full"].cpu().numpy()[:, 0]
        categories = batch.get("category", None)

        class_token_embedding = None
        if categories is not None:
            cat_name = categories[0] if isinstance(categories, (list, tuple)) else categories
            class_token_embedding = model.get_class_token_embedding(cat_name, device=device)

        with autocast(enabled=config["training"].get("mixed_precision", True)):
            outputs = model(images, class_token_embedding=class_token_embedding)

        outputs_noisy = None
        if use_noisy_val:
            images_noisy = _perturb_for_noisy_calibration(images, eval_cfg, robustness_cfg)
            with autocast(enabled=config["training"].get("mixed_precision", True)):
                outputs_noisy = model(images_noisy, class_token_embedding=class_token_embedding)

        processed_maps = postprocess_anomaly_map(
            torch.sigmoid(outputs["anomaly_map"]),
            target_size=(img_size, img_size),
            sigma=sigma,
            clip_percentiles=clip_percentiles,
            median_ksize=median_ksize,
        )

        if outputs_noisy is not None:
            processed_maps_noisy = postprocess_anomaly_map(
                torch.sigmoid(outputs_noisy["anomaly_map"]),
                target_size=(img_size, img_size),
                sigma=sigma,
                clip_percentiles=clip_percentiles,
                median_ksize=median_ksize,
            )

        all_scores.extend(outputs["anomaly_score"].detach().cpu().numpy().tolist())
        if outputs_noisy is not None:
            all_scores_noisy.extend(outputs_noisy["anomaly_score"].detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_maps.append(processed_maps)
        if outputs_noisy is not None:
            all_maps_noisy.append(processed_maps_noisy)
        all_masks.append(masks_full)

    all_scores = torch.as_tensor(all_scores).numpy()
    all_labels = torch.as_tensor(all_labels).numpy().astype(int)
    all_maps = np.concatenate(all_maps, axis=0)
    all_masks = np.concatenate(all_masks, axis=0).astype(int)

    if use_noisy_val and len(all_scores_noisy) == len(all_scores) and len(all_maps_noisy) == len(all_maps):
        all_scores_noisy = torch.as_tensor(all_scores_noisy).numpy()
        all_maps_noisy = np.concatenate(all_maps_noisy, axis=0)

        image_thr, image_f1_clean, image_f1_noisy = _robust_threshold(
            all_scores,
            all_scores_noisy,
            all_labels,
            num_candidates=int(calib_cfg.get("image_candidates", 256)),
            max_samples=int(calib_cfg.get("image_max_samples", 200000)),
            seed=int(calib_cfg.get("seed", 42)),
        )

        pixel_thr, pixel_f1_clean, pixel_f1_noisy = _robust_threshold(
            all_maps.flatten(),
            all_maps_noisy.flatten(),
            all_masks.flatten(),
            num_candidates=int(calib_cfg.get("pixel_candidates", 128)),
            max_samples=int(calib_cfg.get("pixel_max_samples", 500000)),
            seed=int(calib_cfg.get("seed", 42)),
        )

        return {
            "image_threshold": float(image_thr),
            "pixel_threshold": float(pixel_thr),
            "val_image_f1": float(image_f1_clean),
            "val_pixel_f1": float(pixel_f1_clean),
            "val_image_f1_noisy": float(image_f1_noisy),
            "val_pixel_f1_noisy": float(pixel_f1_noisy),
            "calibration_mode": "robust_clean_noisy",
        }

    p, r, t = precision_recall_curve(all_labels, all_scores)
    f1 = 2.0 * p * r / (p + r + 1e-8)
    if len(t) > 0:
        best_idx = int(np.argmax(f1[:-1]))
        image_thr = float(t[best_idx])
        image_f1 = float(f1[:-1][best_idx])
    else:
        image_thr = 0.5
        image_f1 = 0.0

    px_scores = all_maps.flatten()
    px_labels = all_masks.flatten()
    p2, r2, t2 = precision_recall_curve(px_labels, px_scores)
    f12 = 2.0 * p2 * r2 / (p2 + r2 + 1e-8)
    if len(t2) > 0:
        best_idx2 = int(np.argmax(f12[:-1]))
        pixel_thr = float(t2[best_idx2])
        pixel_f1 = float(f12[:-1][best_idx2])
    else:
        pixel_thr = 0.5
        pixel_f1 = 0.0

    return {
        "image_threshold": image_thr,
        "pixel_threshold": pixel_thr,
        "val_image_f1": image_f1,
        "val_pixel_f1": pixel_f1,
        "calibration_mode": "clean_only",
    }


# These remain for backward compat but are inactive under default.yaml
# (training.distillation.enabled = false)
def setup_distillation_teacher(config, device):
    """Optional teacher model setup for knowledge distillation."""
    train_cfg = config.get("training", {})
    distill_cfg = train_cfg.get("distillation", {})
    if not bool(distill_cfg.get("enabled", False)):
        return None

    teacher_ckpt = str(distill_cfg.get("teacher_checkpoint", "")).strip()
    if not teacher_ckpt:
        raise ValueError(
            "Distillation is enabled, but training.distillation.teacher_checkpoint is not set."
        )
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    teacher_config_path = str(distill_cfg.get("teacher_config", "")).strip()
    if teacher_config_path:
        teacher_config = load_config(teacher_config_path)
    else:
        teacher_config = config

    print("\nInitializing distillation teacher model...")
    teacher_model = SSVP(teacher_config).to(device)

    teacher_state = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    try:
        teacher_model.load_state_dict(teacher_state["model_state"], strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load teacher checkpoint. Set training.distillation.teacher_config "
            "to the config used by the teacher architecture."
        ) from exc

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_param_info = teacher_model.count_parameters()
    print(f"Distillation teacher checkpoint: {teacher_ckpt}")
    print(f"Teacher trainable parameters (forced frozen): {teacher_param_info['trainable']:,}")
    print("Distillation teacher ready (eval mode, gradients disabled).")

    return teacher_model


def compute_distillation_loss(student_outputs, teacher_outputs, distill_cfg):
    """Compute distillation loss terms from teacher outputs."""
    w_map = float(distill_cfg.get("lambda_map", 0.25))
    w_score = float(distill_cfg.get("lambda_score", 0.10))
    w_feat = float(distill_cfg.get("lambda_feat", 0.0))

    s_map = torch.sigmoid(student_outputs["anomaly_map"])
    t_map = torch.sigmoid(teacher_outputs["anomaly_map"]).detach()
    l_map = F.mse_loss(s_map, t_map)

    s_score = student_outputs["anomaly_score"]
    t_score = teacher_outputs["anomaly_score"].detach()
    l_score = F.mse_loss(s_score, t_score)

    if (
        w_feat > 0
        and "v_syn_global" in teacher_outputs
        and student_outputs["v_syn_global"].shape == teacher_outputs["v_syn_global"].shape
    ):
        l_feat = F.mse_loss(
            student_outputs["v_syn_global"],
            teacher_outputs["v_syn_global"].detach(),
        )
    else:
        l_feat = torch.zeros((), device=s_map.device, dtype=s_map.dtype)

    l_distill = w_map * l_map + w_score * l_score + w_feat * l_feat
    return l_distill, {
        "distill_map": l_map,
        "distill_score": l_score,
        "distill_feat": l_feat,
    }


def setup_optimizer(model, config):
    """
    Create optimizer with dual learning rate groups (§4.2).

    Group 1: Prompt embeddings → lr = 5e-4
    Group 2: VAE + projections + gates → lr = 1e-4
    """
    train_cfg = config["training"]
    param_groups = model.get_trainable_params()

    optimizer_groups = []
    for group in param_groups:
        lr_key = group["lr_key"]
        lr = train_cfg.get(lr_key, 1e-4)
        optimizer_groups.append({
            "params": group["params"],
            "lr": lr,
        })

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=train_cfg["weight_decay"],
    )
    return optimizer


def setup_scheduler(optimizer, config, steps_per_epoch):
    """Cosine annealing with optional warmup."""
    train_cfg = config["training"]
    total_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = train_cfg.get("warmup_epochs", 1) * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                    scaler, device, config, epoch, teacher_model=None):
    """
    Train for one epoch with gradient accumulation and mixed precision.

    Implements Algorithm 3 from the paper.
    """
    model.train()
    train_cfg = config["training"]
    grad_accum_steps = train_cfg["grad_accum_steps"]
    consistency_cfg = train_cfg.get("consistency", {})
    use_consistency = bool(consistency_cfg.get("enabled", False))
    lambda_consistency = float(consistency_cfg.get("lambda", 0.0))
    consistency_noise_std = float(consistency_cfg.get("noise_std", 0.04))
    consistency_every_n = max(1, int(consistency_cfg.get("every_n_steps", 1)))
    consistency_start_epoch = max(1, int(consistency_cfg.get("start_epoch", 1)))
    distill_cfg = train_cfg.get("distillation", {})
    use_distillation = bool(distill_cfg.get("enabled", False)) and teacher_model is not None

    total_loss = 0.0
    loss_components = {
        "seg": 0,
        "class": 0,
        "vae": 0,
        "reg": 0,
        "denoise": 0,
        "consistency": 0,
        "distill": 0,
        "distill_map": 0,
        "distill_score": 0,
        "distill_feat": 0,
    }
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}",
                leave=True, ncols=120)

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.float32, device=device)
        categories = batch.get("category", None)

        class_token_embedding = None
        teacher_class_token_embedding = None
        if categories is not None:
            cat_name = categories[0] if isinstance(categories, (list, tuple)) else categories
            class_token_embedding = model.get_class_token_embedding(cat_name, device=device)
            if use_distillation:
                teacher_class_token_embedding = teacher_model.get_class_token_embedding(
                    cat_name,
                    device=device,
                )

        teacher_outputs = None
        if use_distillation:
            with torch.no_grad():
                with autocast(enabled=train_cfg["mixed_precision"]):
                    teacher_outputs = teacher_model(
                        images,
                        class_token_embedding=teacher_class_token_embedding,
                    )

        # ── Forward pass with mixed precision ──
        with autocast(enabled=train_cfg["mixed_precision"]):
            outputs = model(images, class_token_embedding=class_token_embedding)

            targets = {"mask": masks, "label": labels}
            loss, loss_dict = criterion(outputs, targets)

            if use_distillation and teacher_outputs is not None:
                l_distill, distill_parts = compute_distillation_loss(
                    outputs,
                    teacher_outputs,
                    distill_cfg,
                )
                loss = loss + l_distill
                loss_dict["distill"] = float(l_distill.detach().item())
                loss_dict["distill_map"] = float(distill_parts["distill_map"].detach().item())
                loss_dict["distill_score"] = float(distill_parts["distill_score"].detach().item())
                loss_dict["distill_feat"] = float(distill_parts["distill_feat"].detach().item())
                loss_dict["total"] = float(loss.detach().item())
            else:
                loss_dict["distill"] = 0.0
                loss_dict["distill_map"] = 0.0
                loss_dict["distill_score"] = 0.0
                loss_dict["distill_feat"] = 0.0

            should_consistency = (
                use_consistency
                and lambda_consistency > 0
                and (epoch + 1) >= consistency_start_epoch
                and (batch_idx % consistency_every_n == 0)
            )

            if should_consistency:
                images_noisy = _perturb_for_consistency(images, noise_std=consistency_noise_std)
                with torch.no_grad():
                    outputs_noisy = model(images_noisy, class_token_embedding=class_token_embedding)
                l_cons = F.mse_loss(outputs["v_syn_global"], outputs_noisy["v_syn_global"].detach())
                loss = loss + lambda_consistency * l_cons
                loss_dict["consistency"] = float(l_cons.detach().item())
                loss_dict["total"] = float(loss.detach().item())
            else:
                loss_dict["consistency"] = 0.0

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        # ── Backward pass ──
        scaler.scale(loss).backward()

        # ── Optimizer step (every grad_accum_steps) ──
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # ── Logging ──
        total_loss += loss_dict["total"]
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "seg": f"{loss_dict['seg']:.4f}",
            "cls": f"{loss_dict['class']:.4f}",
            "dst": f"{loss_dict['distill']:.4f}",
            "α": f"{model.vcpg.alpha.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Handle remaining gradients
    if num_batches % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Average losses
    avg_loss = total_loss / max(num_batches, 1)
    avg_components = {k: v / max(num_batches, 1) for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """Validate on a held-out split from train data (anti-leakage)."""
    model.eval()

    total_loss = 0.0
    loss_components = {
        "seg": 0.0,
        "class": 0.0,
        "vae": 0.0,
        "reg": 0.0,
        "denoise": 0.0,
    }
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating(train-split)", leave=False, ncols=100):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.float32, device=device)
        categories = batch.get("category", None)

        class_token_embedding = None
        if categories is not None:
            cat_name = categories[0] if isinstance(categories, (list, tuple)) else categories
            class_token_embedding = model.get_class_token_embedding(cat_name, device=device)

        with autocast(enabled=config["training"]["mixed_precision"]):
            outputs = model(images, class_token_embedding=class_token_embedding)
            targets = {"mask": masks, "label": labels}
            loss, loss_dict = criterion(outputs, targets)

        total_loss += float(loss_dict["total"])
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += float(loss_dict[key])
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_components = {k: v / max(num_batches, 1) for k, v in loss_components.items()}

    return {
        "val_loss": avg_loss,
        "val_seg": avg_components["seg"],
        "val_class": avg_components["class"],
        "val_vae": avg_components["vae"],
        "val_reg": avg_components["reg"],
        "val_denoise": avg_components.get("denoise", 0.0),
    }


@torch.no_grad()
def save_validation_visualizations(model, val_loader, device, config, output_dir, num_images=5):
    """Save anomaly-map overlays for the first N validation images."""
    model.eval()
    saved = 0

    vis_dir = os.path.join(output_dir, "val_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for old_name in os.listdir(vis_dir):
        old_path = os.path.join(vis_dir, old_name)
        if os.path.isfile(old_path):
            os.remove(old_path)

    sigma = config.get("eval", {}).get("gaussian_sigma", 1.5)
    img_size = config["data"]["img_size"]

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks_full = batch.get("mask_full")
        categories = batch.get("category", None)

        class_token_embedding = None
        if categories is not None:
            cat_name = categories[0] if isinstance(categories, (list, tuple)) else categories
            class_token_embedding = model.get_class_token_embedding(cat_name, device=device)

        with autocast(enabled=config["training"]["mixed_precision"]):
            outputs = model(images, class_token_embedding=class_token_embedding)

        processed_maps = postprocess_anomaly_map(
            torch.sigmoid(outputs["anomaly_map"]),
            target_size=(img_size, img_size),
            sigma=sigma,
        )

        batch_size = images.shape[0]
        for i in range(batch_size):
            if saved >= num_images:
                return saved

            img_np = denormalize_image(images[i].cpu())
            score = float(torch.as_tensor(outputs["anomaly_score"]).detach().flatten()[i].item())

            mask = None
            if masks_full is not None:
                mask = masks_full[i, 0].cpu().numpy()

            save_path = os.path.join(vis_dir, f"sample_{saved}.png")
            visualize_results(
                img_np,
                processed_maps[i],
                mask=mask,
                save_path=save_path,
                title=f"Validation(train-only) | Score: {score:.3f}",
            )
            saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="SSVP Training")
    parser.add_argument("--config", type=str, default=default_config_path(),
                        help="Path to config file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Specific categories to train on (default: all)")
    parser.add_argument("--output_dir", type=str, default=str(LOGS_DIR / "train_run"),
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Load config ──
    config = load_config(args.config)
    if args.data_root:
        config["data"]["data_root"] = args.data_root

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device setup (strict GPU-only training) ──
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This training run requires GPU only.")

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data loaders (train-only split, no test leakage) ──
    print("\nLoading train/validation datasets (train split only)...")
    train_loader, val_loader = get_mvtec_train_val_dataloaders(
        config,
        source_categories=args.categories,
        seed=args.seed,
    )
    from data.mvtec import log_dataset_statistics
    train_stats = log_dataset_statistics(train_loader.dataset, "Train Split")
    val_stats = log_dataset_statistics(val_loader.dataset, "Validation Split")

    # Save to output_dir for P-PRO diagnostics
    import json
    stats_path = os.path.join(args.output_dir, "dataset_statistics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"train": train_stats, "val": val_stats}, f, indent=2)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples (from train split): {len(val_loader.dataset)}")

    # ── Model ──
    print("\nInitializing SSVP model...")
    model = SSVP(config).to(device)
    param_info = model.count_parameters()
    print(f"Total parameters:     {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Frozen parameters:    {param_info['frozen']:,}")
    if getattr(model, "pruning_info", {}).get("enabled", False):
        pruning_info = model.pruning_info
        print(f"Head pruning enabled: mode={pruning_info.get('mode')}")
        if "pruned_layers" in pruning_info:
            print(f"Pruned layers by scope: {pruning_info.get('pruned_layers')}")
        if "depth_kept_layers" in pruning_info and "depth_total_layers" in pruning_info:
            print(
                "Depth pruning layers kept: "
                f"{pruning_info.get('depth_kept_layers')}/{pruning_info.get('depth_total_layers')}"
            )
    if getattr(model, "lora_info", {}).get("enabled", False):
        print(
            "LoRA enabled: "
            f"wrapped={model.lora_info.get('wrapped_linears', 0)}, "
            f"rank={model.lora_info.get('rank')}, "
            f"alpha={model.lora_info.get('alpha')}, "
            f"dropout={model.lora_info.get('dropout')}"
        )

    # ── Optimizer, Scheduler, Criterion ──
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))
    criterion = SSVPLoss(config).to(device)
    scaler = GradScaler(enabled=config["training"]["mixed_precision"])
    teacher_model = setup_distillation_teacher(config, device)

    # ── Resume from checkpoint ──
    start_epoch = 0
    best_auroc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_auroc = checkpoint.get("best_auroc", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ──
    train_cfg = config["training"]
    early_stop_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(early_stop_cfg.get("enabled", True))
    es_patience = int(early_stop_cfg.get("patience", 5))
    es_min_delta = float(early_stop_cfg.get("min_delta", 1e-4))

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    metrics_history = []

    print(f"\n{'='*60}")
    print(f"  SSVP Training — {train_cfg['epochs']} epochs")
    print(f"  Batch size: {train_cfg['batch_size']} × {train_cfg['grad_accum_steps']} "
          f"accum = {train_cfg['batch_size'] * train_cfg['grad_accum_steps']} effective")
    print(f"  Mixed precision: {train_cfg['mixed_precision']}")
    print(f"  Early stopping: {es_enabled} (patience={es_patience}, min_delta={es_min_delta})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()

        # ── Train ──
        avg_loss, loss_components = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, config, epoch, teacher_model=teacher_model
        )
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']} — "
              f"Loss: {avg_loss:.4f} | "
              f"Seg: {loss_components['seg']:.4f} | "
              f"Cls: {loss_components['class']:.4f} | "
              f"Distill: {loss_components['distill']:.4f} | "
              f"VAE: {loss_components['vae']:.4f} | "
              f"Reg: {loss_components['reg']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # ── Validate every epoch on held-out train subset ──
        val_metrics = validate(model, val_loader, criterion, device, config)
        print(f"  Validation(train-only) — "
              f"Loss: {val_metrics['val_loss']:.4f} | "
              f"Seg: {val_metrics['val_seg']:.4f} | "
              f"Cls: {val_metrics['val_class']:.4f} | "
              f"VAE: {val_metrics['val_vae']:.4f} | "
              f"Reg: {val_metrics['val_reg']:.4f} | "
              f"Denoise: {val_metrics['val_denoise']:.4f}")

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "train_seg": float(loss_components["seg"]),
            "train_class": float(loss_components["class"]),
            "train_vae": float(loss_components["vae"]),
            "train_reg": float(loss_components["reg"]),
            "train_denoise": float(loss_components["denoise"]),
            "train_consistency": float(loss_components["consistency"]),
            "train_distill": float(loss_components["distill"]),
            "train_distill_map": float(loss_components["distill_map"]),
            "train_distill_score": float(loss_components["distill_score"]),
            "train_distill_feat": float(loss_components["distill_feat"]),
            **{k: float(v) for k, v in val_metrics.items()},
            "lr": current_lr,
            "epoch_time_sec": float(epoch_time),
        })

        improved = val_metrics["val_loss"] < (best_val_loss - es_min_delta)
        if improved:
            best_val_loss = val_metrics["val_loss"]
            epochs_without_improvement = 0
            best_auroc = max(0.0, 100.0 - best_val_loss)

            save_checkpoint(
                model,
                optimizer,
                epoch,
                {
                    "val_loss": best_val_loss,
                    "train_loss": avg_loss,
                    "monitor": "val_loss",
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            saved = save_validation_visualizations(
                model,
                val_loader,
                device,
                config,
                args.output_dir,
                num_images=5,
            )
            print(f"  ★ New best model saved (val_loss={best_val_loss:.4f}); saved {saved} validation visualizations")
        else:
            epochs_without_improvement += 1
            print(f"  No val_loss improvement for {epochs_without_improvement} epoch(s)")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, {"val_loss": best_val_loss},
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
            )

        if es_enabled and epochs_without_improvement >= es_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    summary = {
        "best_val_loss": float(best_val_loss),
        "epochs_completed": len(metrics_history),
        "dataset_statistics": {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "train_stats": train_stats,
            "val_stats": val_stats,
        },
        "early_stopping_enabled": es_enabled,
        "early_stopping_patience": es_patience,
        "monitor": "val_loss",
        "metrics_file": metrics_path,
        "val_visualizations_dir": os.path.join(args.output_dir, "val_visualizations"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export validation-derived calibration thresholds from best checkpoint.
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"], strict=False)

    calibration = derive_validation_thresholds(model, val_loader, device, config)
    calibration_path = os.path.join(args.output_dir, "calibration_thresholds.json")
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val_loss: {best_val_loss:.4f}")
    print(f"  Metrics saved to: {metrics_path}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Calibration thresholds saved to: {calibration_path}")
    print(f"{'='*60}")


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    # Only save trainable parameters to reduce file size
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(p.data_ptr() == v.data_ptr()
               for p in model.parameters() if p.requires_grad)
    }

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "best_auroc": metrics.get("auroc", 0.0),
    }, path)


if __name__ == "__main__":
    main()
