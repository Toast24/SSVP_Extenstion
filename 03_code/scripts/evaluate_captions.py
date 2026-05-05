"""
Evaluate captioning quality on N test images with BERTScore.

Usage:
    python scripts/evaluate_captions.py \
        --checkpoint best_model.pth \
        --config configs/default.yaml \
        --data_root 04_data/datasets/cable_resplit \
        --categories cable \
        --num_images 50 \
        --output_dir 05_results/caption_eval \
        --cross_domain_categories leather
"""

import argparse
import json
import os
import random
import sys
import copy
from pathlib import Path

import torch
from PIL import Image

from path_utils import LOGS_DIR, default_config_path
from run_full_pipeline import generate_captions, _load_rgb_image
from utils import load_config, set_seed
from data.mvtec import MVTecDataset

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

def evaluate_int8_degradation(images, meta, config, device, output_dir):
    """Compare FP32 vs INT8 captions using BERTScore."""
    print("\n--- Generating FP32 Captions ---")
    config_fp32 = copy.deepcopy(config)
    config_fp32.setdefault("captioning", {})["quantize_text_transformer_int8"] = False
    captions_fp32 = generate_captions(images, meta, device, config_fp32)

    print("\n--- Generating INT8 Captions ---")
    config_int8 = copy.deepcopy(config)
    config_int8.setdefault("captioning", {})["quantize_text_transformer_int8"] = True
    captions_int8 = generate_captions(images, meta, device, config_int8)

    print("\n--- Computing BERTScore ---")
    degradation = compute_bertscore(captions_int8, captions_fp32)
    return {
        "fp32_captions": captions_fp32,
        "int8_captions": captions_int8,
        "degradation_bertscore": degradation
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--categories", nargs="+", default=["cable"])
    parser.add_argument("--cross_domain_categories", nargs="+", default=None)
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Source Domain Test Set
    test_ds = MVTecDataset(
        data_root=args.data_root,
        categories=args.categories,
        split="test",
        img_size=config["data"]["img_size"],
        mask_size=config["data"]["mask_size"],
        augment=False
    )
    
    n = len(test_ds)
    if n == 0:
        print("No test images found.")
        sys.exit(1)

    rng = random.Random(args.seed)
    indices = rng.sample(range(n), min(args.num_images, n))

    images_pil = []
    meta = []
    for idx in indices:
        item = test_ds[idx]
        img_np = test_ds.transform.denormalize(item["image"]) if hasattr(test_ds.transform, 'denormalize') else item["image"].permute(1, 2, 0).numpy()
        pil = Image.fromarray((img_np * 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8'))
        images_pil.append(pil)
        meta.append({"category": item["category"], "defect_type": item["defect_type"]})

    # Evaluate INT8 Degradation
    results = evaluate_int8_degradation(images_pil, meta, config, device, args.output_dir)
    
    # Save base results
    with open(os.path.join(args.output_dir, "int8_degradation_report.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Optional: Cross Domain Evaluation
    if args.cross_domain_categories:
        cross_ds = MVTecDataset(
            data_root=args.data_root,
            categories=args.cross_domain_categories,
            split="test",
            img_size=config["data"]["img_size"],
            mask_size=config["data"]["mask_size"],
            augment=False
        )
        cross_n = len(cross_ds)
        cross_indices = rng.sample(range(cross_n), min(args.num_images, cross_n))
        cross_images_pil = []
        cross_meta = []
        for idx in cross_indices:
            item = cross_ds[idx]
            img_np = cross_ds.transform.denormalize(item["image"]) if hasattr(cross_ds.transform, 'denormalize') else item["image"].permute(1, 2, 0).numpy()
            pil = Image.fromarray((img_np * 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8'))
            cross_images_pil.append(pil)
            cross_meta.append({"category": item["category"], "defect_type": item["defect_type"]})
            
        print("\n--- Generating Cross-Domain Captions ---")
        config_cross = copy.deepcopy(config)
        # Assuming run_full_pipeline's generate_captions takes categories list
        cross_captions = generate_captions(cross_images_pil, cross_meta, device, config_cross, categories=args.categories) # Force source prompts
        
        cross_results = {
            "source_prompts_used": args.categories,
            "target_domain": args.cross_domain_categories,
            "cross_domain_captions": cross_captions
        }
        with open(os.path.join(args.output_dir, "cross_domain_report.json"), "w") as f:
            json.dump(cross_results, f, indent=2)

if __name__ == "__main__":
    main()
