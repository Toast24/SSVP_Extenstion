"""
Run full pipeline: train, test, and generate visualizations + captions

Usage (example):
    python 03_code/scripts/run_full_pipeline.py --config 03_code/configs/default.yaml --data_root 04_data/datasets/cable_resplit/cable --output_dir 05_results/ablations/run1

This script:
  - launches `train.py` (on GPU if available)
  - runs `inference.py` to compute metrics
    - samples 5 random test images, generates visualizations and captions
  - saves visualizations and captions under `<output_dir>/visualizations_with_captions/`

Notes:
  - Captioning requires `transformers` and the BLIP model; if not installed the script will print instructions.
    - Captioning supports optional domain fine-tuning and INT8 dynamic quantization on the text transformer.
  - Training is invoked as a subprocess to use the same project scripts.
"""

import os
import sys
import argparse
import subprocess
import random
import json
from pathlib import Path

import yaml

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from path_utils import LOGS_DIR, SCRIPTS_DIR, default_config_path, ensure_import_paths

ensure_import_paths()


def run_subprocess(args, env=None):
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


TRAIN_SCRIPT = str(SCRIPTS_DIR / "train.py")
INFERENCE_SCRIPT = str(SCRIPTS_DIR / "inference.py")
NOISE_EVAL_SCRIPT = str(SCRIPTS_DIR / "evaluate_noise_robustness.py")


def _normalize_token_label(value):
    return str(value or "").strip().lower().replace("_", " ")


def _describe_mask_location(mask_path):
    if not mask_path:
        return ""

    mask_file = Path(mask_path)
    if not mask_file.exists():
        return ""

    with Image.open(mask_file) as mask_img:
        mask = np.array(mask_img.convert("L"))

    foreground = mask > 0
    if not np.any(foreground):
        return ""

    ys, xs = np.where(foreground)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    height, width = mask.shape[:2]

    center_x = ((x_min + x_max) / 2.0) / max(1.0, float(width - 1))
    center_y = ((y_min + y_max) / 2.0) / max(1.0, float(height - 1))
    area_ratio = float(foreground.mean())

    if area_ratio >= 0.40:
        return "covering most of the surface"

    if center_y < 0.33:
        vertical = "upper"
    elif center_y < 0.66:
        vertical = "middle"
    else:
        vertical = "lower"

    if center_x < 0.33:
        horizontal = "left"
    elif center_x < 0.66:
        horizontal = "center"
    else:
        horizontal = "right"

    if horizontal == "center" and vertical == "middle":
        return "near the center"
    if horizontal == "center":
        return f"on the {vertical} part"
    if vertical == "middle":
        return f"on the {horizontal} side"
    return f"on the {vertical} {horizontal} side"


def _build_domain_caption(category, defect_type, mask_path=None):
    category_label = _normalize_token_label(category) or "object"
    defect_label = _normalize_token_label(defect_type) or "good"
    location_label = _describe_mask_location(mask_path)

    if defect_label == "good":
        return f"close-up industrial inspection image of {category_label} with no visible defect"
    if location_label:
        return (
            f"close-up industrial inspection image of {category_label} showing {defect_label} defect "
            f"{location_label}"
        )
    return f"close-up industrial inspection image of {category_label} showing {defect_label} defect"


def _build_generation_prompt(sample, fallback_prompt):
    category = sample.get("category", "object")
    defect = sample.get("defect_type", "good")
    domain_prompt = _build_domain_caption(category, defect)
    if fallback_prompt:
        return f"{fallback_prompt.strip()}: {domain_prompt}"
    return domain_prompt


def _collect_caption_finetune_records(config, categories, max_samples=96, seed=42):
    from data.mvtec import MVTecDataset

    ds = MVTecDataset(
        data_root=config["data"]["data_root"],
        categories=categories,
        split="train",
        img_size=config["data"]["img_size"],
        mask_size=config["data"]["mask_size"],
        augment=False,
        train_all_types=True,
    )

    samples = list(ds.samples)
    if not samples:
        return []

    rng = random.Random(seed)
    rng.shuffle(samples)
    selected = samples[: max(1, int(max_samples))]

    records = []
    for sample in selected:
        records.append(
            {
                "image_path": sample["image_path"],
                "caption": _build_domain_caption(
                    sample.get("category"),
                    sample.get("defect_type"),
                    sample.get("mask_path"),
                ),
            }
        )
    return records


def _load_rgb_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def _finetune_caption_text_model(cap_model, processor, records, device, finetune_cfg):
    epochs = max(1, int(finetune_cfg.get("epochs", 1)))
    batch_size = max(1, int(finetune_cfg.get("batch_size", 4)))
    lr = float(finetune_cfg.get("lr", 2.0e-5))
    weight_decay = float(finetune_cfg.get("weight_decay", 0.0))
    grad_accum_steps = max(1, int(finetune_cfg.get("grad_accum_steps", 1)))
    max_length = max(8, int(finetune_cfg.get("max_length", 40)))
    grad_clip_norm = float(finetune_cfg.get("grad_clip_norm", 1.0))
    seed = int(finetune_cfg.get("seed", 42))

    if not hasattr(cap_model, "text_decoder"):
        print("Caption fine-tuning skipped: model has no text_decoder module.")
        return {"enabled": True, "skipped": True, "reason": "missing_text_decoder"}

    for param in cap_model.parameters():
        param.requires_grad = False
    for param in cap_model.text_decoder.parameters():
        param.requires_grad = True

    trainable_params = [p for p in cap_model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Caption fine-tuning skipped: no trainable parameters after freezing.")
        return {"enabled": True, "skipped": True, "reason": "no_trainable_params"}

    cap_model.train()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    rng = random.Random(seed)
    epoch_losses = []

    for epoch_idx in range(epochs):
        rng.shuffle(records)
        running_loss = 0.0
        n_steps = 0
        accum = 0
        optimizer.zero_grad(set_to_none=True)

        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            batch_images = [_load_rgb_image(r["image_path"]) for r in batch]
            batch_texts = [r["caption"] for r in batch]

            inputs = processor(
                images=batch_images,
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            labels = inputs["input_ids"].clone()
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            outputs = cap_model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=labels,
            )

            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += float(outputs.loss.detach().item())
            n_steps += 1
            accum += 1

            if accum >= grad_accum_steps:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

        if accum > 0:
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / max(1, n_steps)
        epoch_losses.append(avg_loss)
        print(f"Caption fine-tune epoch {epoch_idx + 1}/{epochs} - loss: {avg_loss:.4f}")

    cap_model.eval()
    return {
        "enabled": True,
        "skipped": False,
        "records": len(records),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "epoch_losses": epoch_losses,
    }


def _apply_int8_dynamic_quantization_to_caption_text_transformer(cap_model):
    import torch.ao.quantization as tq

    try:
        if hasattr(cap_model, "text_decoder"):
            text_decoder = cap_model.text_decoder
            if hasattr(text_decoder, "bert") and hasattr(text_decoder.bert, "encoder"):
                text_decoder.bert.encoder = tq.quantize_dynamic(
                    text_decoder.bert.encoder,
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                return True, "text_decoder.bert.encoder"

            if hasattr(text_decoder, "transformer"):
                text_decoder.transformer = tq.quantize_dynamic(
                    text_decoder.transformer,
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                return True, "text_decoder.transformer"

            cap_model.text_decoder = tq.quantize_dynamic(
                cap_model.text_decoder,
                {nn.Linear},
                dtype=torch.qint8,
            )
            return True, "text_decoder"

        if hasattr(cap_model, "transformer"):
            cap_model.transformer = tq.quantize_dynamic(
                cap_model.transformer,
                {nn.Linear},
                dtype=torch.qint8,
            )
            return True, "transformer"
    except Exception as exc:
        print(f"INT8 quantization failed for caption text transformer: {exc}")

    return False, None


def generate_captions(images_pil, meta, device, config, output_dir=None, seed=42, categories=None):
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except Exception as e:
        print("transformers not available. Install with: pip install -r requirements.txt")
        raise

    caption_cfg = config.get("captioning", {})
    model_name = caption_cfg.get("model_name", "Salesforce/blip-image-captioning-base")
    max_length = int(caption_cfg.get("max_length", 32))
    num_beams = int(caption_cfg.get("num_beams", 4))
    base_prompt = str(caption_cfg.get("prompt", ""))
    use_generation_prompt = bool(
        caption_cfg.get("use_generation_prompt", caption_cfg.get("use_domain_prompt", False))
    )
    use_int8 = bool(caption_cfg.get("quantize_text_transformer_int8", False))

    finetune_cfg = caption_cfg.get("domain_finetune", {})
    finetune_enabled = bool(finetune_cfg.get("enabled", False))

    processor = BlipProcessor.from_pretrained(model_name)
    cap_model = BlipForConditionalGeneration.from_pretrained(model_name)

    finetune_summary = {"enabled": False, "skipped": True, "reason": "disabled"}
    if finetune_enabled:
        records = _collect_caption_finetune_records(
            config=config,
            categories=categories,
            max_samples=int(finetune_cfg.get("max_train_samples", 96)),
            seed=int(finetune_cfg.get("seed", seed)),
        )
        if records:
            cap_model = cap_model.to(device)
            finetune_summary = _finetune_caption_text_model(
                cap_model=cap_model,
                processor=processor,
                records=records,
                device=device,
                finetune_cfg=finetune_cfg,
            )
        else:
            print("Caption fine-tuning skipped: no dataset samples were found.")
            finetune_summary = {"enabled": True, "skipped": True, "reason": "no_records"}

    caption_device = device
    if use_int8:
        cap_model = cap_model.to(torch.device("cpu"))
        quantized, target = _apply_int8_dynamic_quantization_to_caption_text_transformer(cap_model)
        if quantized:
            caption_device = torch.device("cpu")
            print(f"Applied INT8 dynamic quantization to caption model {target}.")
        else:
            cap_model = cap_model.to(device)
            caption_device = device

    cap_model.eval()

    if output_dir:
        report_path = Path(output_dir) / "caption_finetune_summary.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(finetune_summary, f, indent=2)

    captions = []
    for idx, img in enumerate(images_pil):
        sample = meta[idx] if idx < len(meta) else {}
        prompt = base_prompt
        if use_generation_prompt:
            prompt = _build_generation_prompt(sample, base_prompt)

        if prompt:
            inputs = processor(images=img, text=prompt, return_tensors="pt")
        else:
            inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(caption_device) for k, v in inputs.items()}

        out = cap_model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        caption = processor.decode(out[0], skip_special_tokens=True)
        caption = caption.strip()
        if not caption:
            caption = _build_domain_caption(
                sample.get("category", "object"),
                sample.get("defect_type", "good"),
                sample.get("mask_path"),
            )
        captions.append(caption)

    return captions


def sample_and_save_visuals(
    checkpoint,
    config_path,
    data_root,
    output_dir,
    num_samples=5,
    seed=42,
    force_caption_finetune=False,
    force_caption_int8=False,
):
    # Import project utilities to reuse preprocessing and visualization
    from utils import load_config, set_seed, denormalize_image, postprocess_anomaly_map, visualize_results
    from models.ssvp import SSVP
    from data.mvtec import MVTecDataset

    config = load_config(config_path)
    sigma = config.get("eval", {}).get("gaussian_sigma", 1.5)
    categories = None
    if data_root:
        dr = Path(data_root)
        if (dr / "train").is_dir() or (dr / "test").is_dir():
            config["data"]["data_root"] = str(dr.parent)
            categories = [dr.name]
        else:
            config["data"]["data_root"] = data_root

    if force_caption_finetune:
        config.setdefault("captioning", {}).setdefault("domain_finetune", {})["enabled"] = True
    if force_caption_int8:
        config.setdefault("captioning", {})["quantize_text_transformer_int8"] = True

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SSVP(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Build test dataset
    test_ds = MVTecDataset(
        data_root=config["data"]["data_root"],
        categories=categories,
        split="test",
        img_size=config["data"]["img_size"],
        mask_size=config["data"]["mask_size"],
    )

    n = len(test_ds)
    if n == 0:
        print("No test images found.")
        return

    rng = random.Random(seed)
    indices = rng.sample(range(n), min(num_samples, n))

    images_pil = []
    meta = []
    for idx in indices:
        item = test_ds[idx]
        # item: dict with 'image' tensor, 'mask_full', 'label', 'category', 'image_path'
        img_tensor = item["image"]
        img_np = denormalize_image(img_tensor)
        pil = Image.fromarray(img_np)
        images_pil.append(pil)
        meta.append(item)

    # Generate captions (try to use BLIP)
    try:
        captions = generate_captions(
            images_pil,
            meta,
            device,
            config,
            output_dir=output_dir,
            seed=seed,
            categories=categories,
        )
    except Exception:
        captions = [f"Image from {m['category']}" for m in meta]

    # Run model forward on each image and save visualizations with caption overlay saved as separate text
    vis_dir = Path(output_dir) / "visualizations_with_captions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(meta):
        img_tensor = item["image"].unsqueeze(0).to(device)
        class_token_embedding = model.get_class_token_embedding(item.get("category", None), device=device)
        with torch.no_grad():
            outputs = model(img_tensor, class_token_embedding=class_token_embedding)

        anomaly_map = outputs["anomaly_map"]  # [1,1,H',W']
        proc_map = postprocess_anomaly_map(
            anomaly_map,
            target_size=(config["data"]["img_size"], config["data"]["img_size"]),
            sigma=sigma,
        )[0]

        img_np = denormalize_image(item["image"])  # H,W,3
        save_path = vis_dir / f"sample_{i}.png"
        score = float(torch.as_tensor(outputs["anomaly_score"]).detach().flatten()[0].item())
        visualize_results(img_np, proc_map, mask=item.get("mask_full", None)[0] if "mask_full" in item else None,
                  save_path=str(save_path), title=f"Score: {score:.3f}\nCaption: {captions[i]}")

        # Save caption as .txt next to image
        txt_path = vis_dir / f"sample_{i}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(captions[i])

    print(f"Saved {len(meta)} visualizations + captions to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(LOGS_DIR / "run_full_pipeline"))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint path to use when --skip_train is set")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for number of epochs")
    parser.add_argument("--num_vis_samples", type=int, default=5,
                        help="Number of random test images to visualize with captions")
    parser.add_argument("--caption_finetune", action="store_true",
                        help="Enable domain fine-tuning for caption model before generation")
    parser.add_argument("--caption_text_int8", action="store_true",
                        help="Force INT8 dynamic quantization for caption text transformer")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip inference/noise benchmark and only export sampled visuals + captions")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Optionally create a temp config with overridden epochs
    config_for_run = args.config
    if args.epochs is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("training", {})
        cfg["training"]["epochs"] = int(args.epochs)
        temp_cfg_path = Path(output_dir) / "_temp_config.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        config_for_run = str(temp_cfg_path)

    # 1) Train
    if not args.skip_train:
        train_cmd = [sys.executable, TRAIN_SCRIPT, "--config", config_for_run, "--output_dir", output_dir]
        if args.data_root:
            dr = Path(args.data_root)
            # If user pointed to a category directory (e.g., ./cable/), pass its parent as data_root and the folder name as category
            if (dr / "train").is_dir() or (dr / "test").is_dir():
                train_cmd += ["--data_root", str(dr.parent)]
                train_cmd += ["--categories", dr.name]
            else:
                train_cmd += ["--data_root", args.data_root]
        if args.resume:
            train_cmd += ["--resume", args.resume]
        # Allow user to override epochs via env var or pass-through config
        run_subprocess(train_cmd)
    else:
        print("Skipping training as requested (--skip_train).")

    # 2) Resolve checkpoint
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = Path(output_dir) / "best_model.pth"
        if not ckpt.exists():
            # try checkpoint_epoch*.pth fallback
            candidates = list(Path(output_dir).glob("checkpoint_epoch*.pth"))
            if candidates:
                ckpt = sorted(candidates)[-1]

    if not ckpt.exists():
        print("No checkpoint found. Provide --checkpoint or ensure output_dir contains saved checkpoints.")
        sys.exit(1)

    # 3) Evaluate via inference.py + noise robustness (optional)
    if not args.skip_eval:
        eval_dir = Path(output_dir) / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = [sys.executable, INFERENCE_SCRIPT, "--config", config_for_run, "--checkpoint", str(ckpt), "--output_dir", str(eval_dir), "--visualize"]
        calibration_file = Path(output_dir) / "calibration_thresholds.json"
        if calibration_file.exists():
            eval_cmd += ["--calibration_file", str(calibration_file)]
        if args.data_root:
            dr = Path(args.data_root)
            if (dr / "train").is_dir() or (dr / "test").is_dir():
                eval_cmd += ["--data_root", str(dr.parent)]
                eval_cmd += ["--categories", dr.name]
            else:
                eval_cmd += ["--data_root", args.data_root]
        run_subprocess(eval_cmd)

        # 3b) Robustness benchmark on heavily noised test images
        noise_eval_dir = Path(output_dir) / "noise_eval"
        noise_cmd = [
            sys.executable,
            NOISE_EVAL_SCRIPT,
            "--config", config_for_run,
            "--checkpoint", str(ckpt),
            "--output_dir", str(noise_eval_dir),
            "--clean_results", str(eval_dir / "results.json"),
        ]
        if calibration_file.exists():
            noise_cmd += ["--calibration_file", str(calibration_file)]

        if args.data_root:
            dr = Path(args.data_root)
            if (dr / "train").is_dir() or (dr / "test").is_dir():
                noise_cmd += ["--data_root", str(dr.parent)]
                noise_cmd += ["--categories", dr.name]
            else:
                noise_cmd += ["--data_root", args.data_root]

        run_subprocess(noise_cmd)

    # 4) Sample random test images, generate captions + visualizations
    sample_and_save_visuals(
        str(ckpt), config_for_run, args.data_root, output_dir,
        num_samples=args.num_vis_samples,
        force_caption_finetune=args.caption_finetune,
        force_caption_int8=args.caption_text_int8,
    )


if __name__ == "__main__":
    main()
