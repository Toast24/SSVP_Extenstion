"""
Evaluate captioning quality on N test images with BERTScore.

This benchmark compares a no-hint baseline captioner against a spatially
fine-tuned captioner on the same test images. References are built from the
existing train/test split using the ground-truth mask location and defect type.
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from path_utils import default_config_path, ensure_import_paths

ensure_import_paths()

from data.mvtec import MVTecDataset
from run_full_pipeline import generate_captions
from caption_utils import _build_caption_target, _mask_to_spatial_location
from utils import load_config, set_seed
from utils import postprocess_anomaly_map, compute_image_level_metrics
from models.ssvp import SSVP
from transformers import pipeline
import re


def compute_bertscore(predictions, references):
    """Compute BERTScore between generated and reference captions."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("bert_score not available. Install with: pip install bert-score>=0.3.13")
        raise

    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "per_sample_f1": [float(f) for f in F1],
    }


def _prepare_caption_config(config, use_hints, finetune_enabled, finetune_epochs, max_train_samples, target_mode):
    caption_cfg = copy.deepcopy(config)
    caption_section = caption_cfg.setdefault("captioning", {})
    caption_section["use_domain_prompt"] = bool(use_hints)
    finetune_section = caption_section.setdefault("domain_finetune", {})
    finetune_section["enabled"] = bool(finetune_enabled)
    finetune_section["epochs"] = int(finetune_epochs)
    finetune_section["target_mode"] = str(target_mode)
    finetune_section["max_train_samples"] = None if max_train_samples is None else int(max_train_samples)
    return caption_cfg


def _build_sample_meta(item):
    return {
        "category": item["category"],
        "defect_type": item["defect_type"],
        "mask_full": item.get("mask_full", None),
        "label": int(item.get("label", 0)),
    }


def _load_test_samples(test_ds, indices):
    images_pil = []
    meta = []
    for idx in indices:
        item = test_ds[idx]
        img_np = test_ds.transform.denormalize(item["image"]) if hasattr(test_ds.transform, "denormalize") else item["image"].permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:
            pil = Image.fromarray((img_np * 255).astype("uint8"))
        else:
            pil = Image.fromarray(img_np.astype("uint8"))
        images_pil.append(pil)
        meta.append(_build_sample_meta(item))
    return images_pil, meta


def _evaluate_variant(images, meta, base_config, device, use_hints, finetune_enabled, finetune_epochs, max_train_samples, target_mode, synthesize_from_mask=False):
    config_variant = _prepare_caption_config(
        config=base_config,
        use_hints=use_hints,
        finetune_enabled=finetune_enabled,
        finetune_epochs=finetune_epochs,
        max_train_samples=max_train_samples,
        target_mode=target_mode,
    )
    # Optionally synthesize deterministic captions from ground-truth masks
    if synthesize_from_mask:
        captions = []
        for sample in meta:
            defect = str(sample.get("defect_type", "")).lower()
            if defect == "good":
                captions.append("no anomaly")
            else:
                loc = _mask_to_spatial_location(sample.get("mask_full", None))
                captions.append(f"anomaly at {loc}")
    else:
        captions = generate_captions(images, meta, device, config_variant)
    references = [_build_caption_target(sample, target_mode=target_mode) for sample in meta]
    score = compute_bertscore(captions, references)

    # Token-level anomaly-presence exact-match and spatial match
    def _norm_token(s):
        return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())

    anomaly_matches = []
    spatial_matches = []
    for pred, ref, sample in zip(captions, references, meta):
        pred_n = _norm_token(pred)
        anomaly_label = "no anomaly" if _norm_token(sample.get("defect_type", "")) == "good" else "anomaly"
        anomaly_matches.append(1 if anomaly_label in pred_n else 0)

        # spatial match: compute expected location from mask and check presence
        expected_loc = _mask_to_spatial_location(sample.get("mask_full", None))
        loc_n = _norm_token(expected_loc)
        if not loc_n or loc_n == "center":
            spatial_ok = 1 if ("center" in pred_n or "middle" in pred_n or "centre" in pred_n) else 0
        else:
            parts = loc_n.split()
            # require at least one of the directional words to appear
            spatial_ok = 1 if any(p in pred_n for p in parts) else 0
        spatial_matches.append(int(spatial_ok))

    # Optional: NLI entailment score (prediction entails reference)
    nli_scores = None
    try:
        nli = pipeline("text-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
        entail_scores = []
        for pred, ref in zip(captions, references):
            out = nli(pred, ref, return_all_scores=True)
            # out: list of dicts with labels; find ENTAILMENT score
            score_ent = 0.0
            for item in out[0]:
                if item["label"].upper() == "ENTAILMENT":
                    score_ent = float(item["score"])
                    break
            entail_scores.append(score_ent)
        nli_scores = entail_scores
    except Exception:
        nli_scores = None

    return {
        "captions": captions,
        "references": references,
        "bertscore": score,
        "anomaly_matches": anomaly_matches,
        "spatial_matches": spatial_matches,
        "nli_entailment": nli_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--categories", nargs="+", default=["cable"])
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption_target_mode", type=str, default="spatial", choices=["spatial", "domain"])
    parser.add_argument("--caption_finetune_epochs", type=int, default=1)
    parser.add_argument("--caption_max_train_samples", type=int, default=0)
    parser.add_argument("--synthesize_from_mask", action="store_true", help="Skip BLIP and synthesize captions deterministically from ground-truth masks")
    parser.add_argument("--synthesize_from_predicted_map", action="store_true", help="Run model inference and synthesize captions from predicted anomaly maps")
    parser.add_argument("--cross_domain_categories", nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    config.setdefault("data", {})["data_root"] = args.data_root
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = MVTecDataset(
        data_root=args.data_root,
        categories=args.categories,
        split="test",
        img_size=config["data"]["img_size"],
        mask_size=config["data"]["mask_size"],
        augment=False,
    )

    n = len(test_ds)
    if n == 0:
        print("No test images found.")
        sys.exit(1)

    rng = random.Random(args.seed)
    indices = rng.sample(range(n), min(args.num_images, n))
    images_pil, meta = _load_test_samples(test_ds, indices)

    max_train_samples = None if args.caption_max_train_samples <= 0 else args.caption_max_train_samples

    image_scores = None
    image_labels = None
    # If requested, run model inference to predict anomaly maps for the selected indices
    if args.synthesize_from_predicted_map:
        print("Running model inference to predict anomaly maps for selected samples...")
        model = SSVP(config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        model.eval()

        sigma = config.get("eval", {}).get("gaussian_sigma", 1.5)
        image_scores = []
        image_labels = []
        for i, idx in enumerate(indices):
            item = test_ds[idx]
            img_tensor = item["image"].unsqueeze(0).to(device)
            cat = item.get("category", None)
            class_token_embedding = model.get_class_token_embedding(cat, device=device) if hasattr(model, "get_class_token_embedding") else None
            with torch.no_grad():
                outputs = model(img_tensor, class_token_embedding=class_token_embedding)
            image_scores.append(float(outputs["anomaly_score"].detach().cpu().flatten()[0].item()))
            image_labels.append(int(item.get("label", 0)))
            anomaly_map = torch.sigmoid(outputs["anomaly_map"])  # [1,1,H',W']
            processed = postprocess_anomaly_map(anomaly_map, target_size=(config["data"]["img_size"], config["data"]["img_size"]), sigma=sigma)[0]
            # binarize at 0.5 for spatial localization
            mask_full = (processed > 0.5).astype("uint8")
            # attach predicted mask to meta so downstream synthesis can use it
            meta[i]["mask_full"] = mask_full

    print("\n--- Baseline no-hint captions ---")
    baseline = _evaluate_variant(
        images=images_pil,
        meta=meta,
        base_config=config,
        device=device,
        use_hints=False,
        finetune_enabled=False,
        finetune_epochs=0,
        max_train_samples=max_train_samples,
        target_mode=args.caption_target_mode,
    )

    print("\n--- BLIP fine-tuned captions ---")
    blip_finetuned = _evaluate_variant(
        images=images_pil,
        meta=meta,
        base_config=config,
        device=device,
        use_hints=False,
        finetune_enabled=True,
        finetune_epochs=args.caption_finetune_epochs,
        max_train_samples=max_train_samples,
        target_mode=args.caption_target_mode,
    )

    print("\n--- Predicted-map synthesis captions ---")
    predicted_map = _evaluate_variant(
        images=images_pil,
        meta=meta,
        base_config=config,
        device=device,
        use_hints=False,
        finetune_enabled=False,
        finetune_epochs=0,
        max_train_samples=max_train_samples,
        target_mode=args.caption_target_mode,
        synthesize_from_mask=args.synthesize_from_predicted_map or args.synthesize_from_mask,
    )

    results = {
        "categories": args.categories,
        "num_images": len(images_pil),
        "target_mode": args.caption_target_mode,
        "baseline": baseline,
        "blip_finetuned": blip_finetuned,
        "predicted_map_synthesis": predicted_map,
    }

    if image_scores is not None and image_labels is not None:
        detection_metrics = compute_image_level_metrics(np.array(image_scores), np.array(image_labels))
        results["image_level_detection"] = detection_metrics

    # Keep caption deltas for compatibility.
    results["delta_f1_blip"] = float(blip_finetuned.get("bertscore", {}).get("f1", 0.0) - baseline.get("bertscore", {}).get("f1", 0.0))
    results["delta_f1_predicted"] = float(predicted_map.get("bertscore", {}).get("f1", 0.0) - baseline.get("bertscore", {}).get("f1", 0.0))

    report_path = os.path.join(args.output_dir, "caption_before_after_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved caption benchmark to: {report_path}")
    if "image_level_detection" in results:
        det = results["image_level_detection"]
        print(
            f"Image AUROC: {det['auroc']:.2f} | F1-Max: {det['f1_max']:.2f} | AP: {det['ap']:.2f}"
        )
        print(
            f"BLIP F1: {blip_finetuned['bertscore']['f1']:.4f} | Predicted-map F1: {predicted_map['bertscore']['f1']:.4f}"
        )
    else:
        print(
            f"Baseline F1: {baseline['bertscore']['f1']:.4f} | "
            f"BLIP F1: {blip_finetuned['bertscore']['f1']:.4f} | "
            f"Predicted-map F1: {predicted_map['bertscore']['f1']:.4f}"
        )

    if args.cross_domain_categories:
        cross_ds = MVTecDataset(
            data_root=args.data_root,
            categories=args.cross_domain_categories,
            split="test",
            img_size=config["data"]["img_size"],
            mask_size=config["data"]["mask_size"],
            augment=False,
        )
        cross_n = len(cross_ds)
        cross_indices = rng.sample(range(cross_n), min(args.num_images, cross_n))
        cross_images_pil, cross_meta = _load_test_samples(cross_ds, cross_indices)
        config_cross = _prepare_caption_config(
            config=config,
            use_hints=False,
            finetune_enabled=True,
            finetune_epochs=args.caption_finetune_epochs,
            max_train_samples=max_train_samples,
            target_mode=args.caption_target_mode,
        )
        cross_captions = generate_captions(cross_images_pil, cross_meta, device, config_cross)
        cross_results = {
            "source_prompts_used": args.categories,
            "target_domain": args.cross_domain_categories,
            "cross_domain_captions": cross_captions,
        }
        with open(os.path.join(args.output_dir, "cross_domain_report.json"), "w", encoding="utf-8") as f:
            json.dump(cross_results, f, indent=2)


if __name__ == "__main__":
    main()
