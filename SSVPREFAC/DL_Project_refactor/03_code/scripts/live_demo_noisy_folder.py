"""
Run a live segmentation + captioning demo on noisy images from a user-specified folder.

Default behavior uses run21 best model and enables caption text INT8 compression.

Example:
    python 03_code/scripts/live_demo_noisy_folder.py \
            --input_folder 04_data/datasets/cable/test/combined \
            --output_dir 05_results/ablations/live_demo_combined
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from path_utils import LOGS_DIR, default_config_path, ensure_import_paths

ensure_import_paths(include_scripts=True)

from data.transforms import DualResizeTransform
from models.ssvp import SSVP
from run_full_pipeline import generate_captions
from utils import load_config, postprocess_anomaly_map, set_seed, visualize_results


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_image_paths(input_folder: Path, recursive: bool):
    if recursive:
        files = [p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    else:
        files = [p for p in input_folder.glob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files)


def infer_defect_hint(image_path: Path, input_folder: Path):
    parent = image_path.parent.name.strip().lower().replace("_", " ")
    root = input_folder.name.strip().lower().replace("_", " ")
    if not parent or parent in {root, "images", "image", "img", "noisy", "test", "dataset"}:
        return "anomaly"
    return parent


def normalize_map(anomaly_map: np.ndarray):
    lo = float(np.percentile(anomaly_map, 1.0))
    hi = float(np.percentile(anomaly_map, 99.0))
    if hi - lo < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return np.clip((anomaly_map - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def sigmoid_prob(logit: float):
    # Clamp logits to keep exp stable while preserving probability shape.
    z = float(np.clip(logit, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def save_segmentation_artifacts(image_np: np.ndarray, map_norm: np.ndarray, mask_bin: np.ndarray, out_base: Path):
    heatmap_uint8 = (map_norm * 255.0).astype(np.uint8)
    mask_uint8 = (mask_bin.astype(np.uint8) * 255)

    Image.fromarray(heatmap_uint8).save(str(out_base.with_suffix(".heatmap.png")))
    Image.fromarray(mask_uint8).save(str(out_base.with_suffix(".mask.png")))

    overlay = image_np.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    blend_idx = mask_bin.astype(bool)
    overlay[blend_idx] = (0.65 * overlay[blend_idx] + 0.35 * red).astype(np.uint8)
    Image.fromarray(overlay).save(str(out_base.with_suffix(".overlay.png")))


def load_model(config, checkpoint_path: Path, device: torch.device):
    model = SSVP(config).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Live demo for noisy-folder segmentation + captioning")
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--checkpoint", type=str, default=str(LOGS_DIR / "run21_resplit_15es" / "best_model.pth"))
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing noisy input images")
    parser.add_argument("--output_dir", type=str, default=str(LOGS_DIR / "live_demo"))
    parser.add_argument("--category", type=str, default="cable")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Optional dataset root/category path for caption fine-tune sampling")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_images", type=int, default=0,
                        help="0 means all images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_threshold", type=float, default=0.50,
                        help="Threshold on sigmoid(anomaly_score) used for image-level anomaly decision")
    parser.add_argument("--pixel_threshold", type=float, default=0.60,
                        help="Pixel threshold over post-processed anomaly probability map for segmentation mask")
    parser.add_argument("--min_defect_area_ratio", type=float, default=0.01,
                        help="Minimum fraction of pixels over pixel_threshold to mark image as anomalous")
    parser.add_argument("--caption_finetune", action="store_true",
                        help="Enable caption domain fine-tuning before generation")
    parser.add_argument("--no_caption_text_int8", action="store_true",
                        help="Disable INT8 text-transformer compression for captions")
    parser.add_argument("--caption_prompt_prefix", type=str, default="industrial inspection photo")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations_with_captions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if args.data_root:
        config.setdefault("data", {})["data_root"] = args.data_root

    caption_cfg = config.setdefault("captioning", {})
    caption_cfg["use_domain_prompt"] = True
    caption_cfg["prompt"] = str(args.caption_prompt_prefix)
    caption_cfg["quantize_text_transformer_int8"] = not bool(args.no_caption_text_int8)
    caption_cfg.setdefault("domain_finetune", {})["enabled"] = bool(args.caption_finetune)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = DualResizeTransform(
        img_size=int(config["data"]["img_size"]),
        mask_size=int(config["data"]["mask_size"]),
    )

    image_paths = collect_image_paths(input_folder, recursive=bool(args.recursive))
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_folder}")

    model = load_model(config, checkpoint_path, device)

    image_np_list = []
    image_pil_list = []
    map_prob_list = []
    map_norm_list = []
    score_list = []
    score_prob_list = []
    pixel_ratio_list = []
    pred_label_list = []
    defect_hint_list = []
    meta = []

    sigma = float(config.get("eval", {}).get("gaussian_sigma", 1.5))
    for p in image_paths:
        pil = Image.open(p).convert("RGB")
        img_tensor, _, _ = transform(pil, mask=None)

        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            class_emb = model.get_class_token_embedding(args.category, device=device)
            outputs = model(x, class_token_embedding=class_emb)

        anomaly_map = torch.sigmoid(outputs["anomaly_map"])
        proc_map = postprocess_anomaly_map(
            anomaly_map,
            target_size=(int(config["data"]["img_size"]), int(config["data"]["img_size"])),
            sigma=sigma,
            clip_percentiles=config.get("eval", {}).get("clip_percentiles", [1.0, 99.0]),
            median_ksize=int(config.get("eval", {}).get("median_ksize", 0)),
        )[0]

        map_prob = np.clip(proc_map, 0.0, 1.0).astype(np.float32)
        map_norm = normalize_map(map_prob)
        score = float(torch.as_tensor(outputs["anomaly_score"]).detach().flatten()[0].item())
        score_prob = sigmoid_prob(score)
        pixel_ratio = float((map_prob >= float(args.pixel_threshold)).mean())
        pred_label = int(
            (score_prob >= float(args.image_threshold))
            or (pixel_ratio >= float(args.min_defect_area_ratio))
        )

        img_np = np.array(pil.resize((int(config["data"]["img_size"]), int(config["data"]["img_size"]))))
        image_np_list.append(img_np)
        image_pil_list.append(Image.fromarray(img_np))
        map_prob_list.append(map_prob)
        map_norm_list.append(map_norm)
        score_list.append(score)
        score_prob_list.append(score_prob)
        pixel_ratio_list.append(pixel_ratio)
        pred_label_list.append(pred_label)

        defect_hint = infer_defect_hint(p, input_folder)
        if pred_label == 0:
            defect_hint = "good"
        defect_hint_list.append(defect_hint)

        meta.append(
            {
                "image_path": str(p),
                "category": args.category,
                "defect_type": defect_hint,
                "label": pred_label,
            }
        )

    try:
        captions = generate_captions(
            image_pil_list,
            meta,
            device,
            config,
            output_dir=str(output_dir),
            seed=int(args.seed),
            categories=[args.category],
        )
    except Exception:
        captions = [
            f"industrial inspection image of {args.category} with {m['defect_type']} condition"
            for m in meta
        ]

    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "input_folder": str(input_folder),
        "output_dir": str(output_dir),
        "category": args.category,
        "caption_text_int8": bool(caption_cfg.get("quantize_text_transformer_int8", False)),
        "caption_finetune": bool(caption_cfg.get("domain_finetune", {}).get("enabled", False)),
        "image_threshold": float(args.image_threshold),
        "pixel_threshold": float(args.pixel_threshold),
        "min_defect_area_ratio": float(args.min_defect_area_ratio),
        "num_images": len(image_paths),
        "samples": [],
    }

    for idx, p in enumerate(image_paths):
        map_prob = map_prob_list[idx]
        map_norm = map_norm_list[idx]
        mask_bin = map_prob >= float(args.pixel_threshold)
        score = score_list[idx]
        score_prob = score_prob_list[idx]
        pixel_ratio = pixel_ratio_list[idx]
        pred_label = pred_label_list[idx]

        stem = p.stem
        out_base = vis_dir / f"{idx:03d}_{stem}"
        save_segmentation_artifacts(image_np_list[idx], map_norm, mask_bin, out_base)

        viz_path = vis_dir / f"{idx:03d}_{stem}.viz.png"
        visualize_results(
            image_np_list[idx],
            map_norm,
            mask=None,
            save_path=str(viz_path),
            title=f"Score(logit)={score:.3f} | Prob={score_prob:.3f} | Mask>thr={pixel_ratio:.3f} | Caption={captions[idx]}",
        )

        caption_path = vis_dir / f"{idx:03d}_{stem}.txt"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(captions[idx])

        summary["samples"].append(
            {
                "image_path": str(p),
                "score_logit": float(score),
                "score_probability": float(score_prob),
                "pixel_positive_ratio": float(pixel_ratio),
                "predicted_label": int(pred_label),
                "defect_hint": defect_hint_list[idx],
                "caption": captions[idx],
                "artifacts": {
                    "viz": str(viz_path),
                    "heatmap": str(out_base.with_suffix(".heatmap.png")),
                    "mask": str(out_base.with_suffix(".mask.png")),
                    "overlay": str(out_base.with_suffix(".overlay.png")),
                    "caption": str(caption_path),
                },
            }
        )

    summary_path = output_dir / "demo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Processed {len(image_paths)} images.")
    print(f"Saved demo artifacts to: {vis_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
