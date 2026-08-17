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

import torch
import torch.nn as nn
import numpy as np
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


def _build_generation_prompt(sample, fallback_prompt):
    category = sample.get("category", "object")
    defect = sample.get("defect_type", "good")
    domain_prompt = _build_domain_caption(category, defect)
    if fallback_prompt:
        return f"{fallback_prompt.strip()}: {domain_prompt}"
    return domain_prompt


def _collect_caption_finetune_records(config, categories, max_samples=96, seed=42, target_mode="spatial"):
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
    if max_samples is None:
        selected = samples
    else:
        selected = samples[: max(1, int(max_samples))]

    records = []
    for sample in selected:
        mask_full = None
        if sample.get("mask_path") and os.path.exists(sample["mask_path"]):
            with Image.open(sample["mask_path"]) as mask_img:
                mask_full = np.array(mask_img.convert("L"))
        records.append(
            {
                "image_path": sample["image_path"],
                "caption": _build_caption_target(
                    {
                        "category": sample.get("category"),
                        "defect_type": sample.get("defect_type"),
                        "mask_full": mask_full,
                    },
                    target_mode=target_mode,
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
    use_domain_prompt = bool(caption_cfg.get("use_domain_prompt", True))
    use_int8 = bool(caption_cfg.get("quantize_text_transformer_int8", False))

    finetune_cfg = caption_cfg.get("domain_finetune", {})
    finetune_enabled = bool(finetune_cfg.get("enabled", False))
    target_mode = str(finetune_cfg.get("target_mode", "spatial")).strip().lower()

    try:
        from transformers.utils import import_utils as _transformers_import_utils
        if hasattr(_transformers_import_utils, "check_torch_load_is_safe"):
            _transformers_import_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

    processor = BlipProcessor.from_pretrained(model_name)
    cap_model = BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)

    finetune_summary = {"enabled": False, "skipped": True, "reason": "disabled"}
    if finetune_enabled:
        max_train_samples = finetune_cfg.get("max_train_samples", 96)
        if max_train_samples is None:
            max_train_samples_arg = None
        else:
            max_train_samples_arg = int(max_train_samples)
        records = _collect_caption_finetune_records(
            config=config,
            categories=categories,
            max_samples=max_train_samples_arg,
            seed=int(finetune_cfg.get("seed", seed)),
            target_mode=target_mode,
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
    if not use_int8:
        cap_model = cap_model.to(device)
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
        prompt = ""
        if use_domain_prompt:
            prompt = _build_generation_prompt(sample, "")

        if prompt:
            inputs = processor(images=img, text=prompt, return_tensors="pt")
        else:
            inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(caption_device) for k, v in inputs.items()}

        out = cap_model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        caption = processor.decode(out[0], skip_special_tokens=True)
        caption = caption.strip()
        if not caption:
            caption = _build_domain_caption(sample.get("category", "object"), sample.get("defect_type", "good"))
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
    categories=None,
):
    # Import project utilities to reuse preprocessing and visualization
    from utils import load_config, set_seed, denormalize_image, postprocess_anomaly_map, visualize_results
    from models.ssvp import SSVP
    from data.mvtec import MVTecDataset

    config = load_config(config_path)
    sigma = config.get("eval", {}).get("gaussian_sigma", 1.5)
    inferred_categories = categories
    if data_root and inferred_categories is None:
        dr = Path(data_root)
        if (dr / "train").is_dir() or (dr / "test").is_dir():
            config["data"]["data_root"] = str(dr.parent)
            inferred_categories = [dr.name]
        else:
            config["data"]["data_root"] = data_root
    elif data_root:
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
        categories=inferred_categories,
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
            categories=inferred_categories,
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


def _run_single_category(args, category, cat_output, cat_data, config_for_run):
    """Run the full pipeline (train + eval + noise + captions) for a single category."""
    os.makedirs(cat_output, exist_ok=True)

    # 1) Train
    if not args.skip_train:
        train_cmd = [sys.executable, TRAIN_SCRIPT, "--config", config_for_run, "--output_dir", cat_output]
        if cat_data:
            train_cmd += ["--data_root", cat_data]
            train_cmd += ["--categories", category]
        if args.resume:
            train_cmd += ["--resume", args.resume]
        run_subprocess(train_cmd)
    else:
        print("Skipping training as requested (--skip_train).")

    # 2) Resolve checkpoint
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = Path(cat_output) / "best_model.pth"
        if not ckpt.exists():
            candidates = list(Path(cat_output).glob("checkpoint_epoch*.pth"))
            if candidates:
                ckpt = sorted(candidates)[-1]

    if not ckpt.exists():
        print(f"No checkpoint found for category {category}. Skipping evaluation.")
        return

    # 3) Evaluate via inference.py + noise robustness (optional)
    if not args.skip_eval:
        eval_dir = Path(cat_output) / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = [sys.executable, INFERENCE_SCRIPT, "--config", config_for_run, "--checkpoint", str(ckpt), "--output_dir", str(eval_dir), "--visualize"]
        calibration_file = Path(cat_output) / "calibration_thresholds.json"
        if calibration_file.exists():
            eval_cmd += ["--calibration_file", str(calibration_file)]
        if cat_data:
            eval_cmd += ["--data_root", cat_data]
            eval_cmd += ["--categories", category]
        run_subprocess(eval_cmd)

        # 3b) Robustness benchmark on heavily noised test images
        noise_eval_dir = Path(cat_output) / "noise_eval"
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

        if cat_data:
            noise_cmd += ["--data_root", cat_data]
            noise_cmd += ["--categories", category]

        run_subprocess(noise_cmd)

    # 4) Sample random test images, generate captions + visualizations
    sample_and_save_visuals(
        str(ckpt), config_for_run, cat_data, cat_output,
        num_samples=args.num_vis_samples,
        force_caption_finetune=args.caption_finetune,
        force_caption_int8=args.caption_text_int8,
        categories=[category]
    )


def _write_aggregated_report(aggregated_results, output_dir):
    report_path = Path(output_dir) / "aggregated_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"\nSaved aggregated cross-category report to: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to run sequentially (e.g., cable bottle leather)")
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

    # Main execution loop
    if args.categories and len(args.categories) > 1:
        aggregated_results = {}
        for category in args.categories:
            cat_output = str(Path(args.output_dir) / category)
            # if user points to parent datasets directory, data_root should just be that
            cat_data = args.data_root
            if args.data_root:
                dr = Path(args.data_root)
                if dr.name == category: # User passed a specific category folder but specified multiple categories - this is weird
                    cat_data = str(dr.parent)
                elif (dr / category).is_dir():
                    cat_data = str(dr)
                else: # Fallback to user provided
                    cat_data = args.data_root

            print(f"\n{'='*60}")
            print(f"  Running category: {category}")
            print(f"{'='*60}")
            
            _run_single_category(args, category, cat_output, cat_data, config_for_run)
            
            # Collect results
            results_path = Path(cat_output) / "eval_results" / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    aggregated_results[category] = json.load(f)

        # Write aggregated cross-category report
        _write_aggregated_report(aggregated_results, args.output_dir)
    else:
        # Single-category path (existing behavior)
        category = args.categories[0] if args.categories else None
        dr = Path(args.data_root) if args.data_root else None
        
        cat_data = args.data_root
        if dr and ((dr / "train").is_dir() or (dr / "test").is_dir()):
            cat_data = str(dr.parent)
            if not category:
                category = dr.name
        
        _run_single_category(args, category, output_dir, cat_data, config_for_run)

if __name__ == "__main__":
    main()
