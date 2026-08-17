"""
SSVP: Synergistic Semantic-Visual Prompting — Main Model Orchestrator

Combines all three modules (HSVS, VCPG, VTAM) into a unified end-to-end model.
Implements the forward inference pipeline described in Algorithm 1 of the paper.

Pipeline:
    1. Dual backbone extracts frozen CLIP + DINOv2 features (multi-scale)
    2. HSVS fuses features via Adaptive Token Features Fusion
    3. VCPG generates vision-conditioned prompts via VAE + cross-attention
    4. VTAM produces anomaly maps via AnomalyMoE and final scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import re
import torch.nn.utils.prune as torch_prune

from .backbones import DualBackbone
from .hsvs import HSVS
from .lora import LoRALinear, apply_lora_to_module
from .vcpg import VCPG
from .vtam import VTAM


class DenoisingAutoencoder(nn.Module):
    """Lightweight U-Net style denoiser operating on normalized images."""

    def __init__(self, base_channels=24, dropout=0.0):
        super().__init__()
        c = int(base_channels)
        p = float(dropout)

        def _block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Dropout2d(p=p) if p > 0.0 else nn.Identity(),
            )

        self.enc1 = _block(3, c, stride=1)
        self.enc2 = _block(c, c * 2, stride=2)
        self.enc3 = _block(c * 2, c * 4, stride=2)
        self.bottleneck = _block(c * 4, c * 4, stride=1)

        self.dec2 = _block(c * 4 + c * 2, c * 2, stride=1)
        self.dec1 = _block(c * 2 + c, c, stride=1)
        self.out = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, noisy_images):
        e1 = self.enc1(noisy_images)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, e1], dim=1))

        pred_noise = self.out(u1)
        return noisy_images - pred_noise


class SSVP(nn.Module):
    """
    Synergistic Semantic-Visual Prompting (SSVP) — Full Framework.

    Frozen components:
        - CLIP Image Encoder (ViT-L/14)
        - DINOv2 Encoder (ViT-L/14)
        - CLIP Text Encoder

    Trainable components:
        - HSVS: ATF projection matrices + MLPs (~13M params)
        - VCPG: Prompt embeddings + VAE + cross-attention + α gate
        - VTAM: Global gate MLP + local gate convs

    Args:
        config: Full configuration dict
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._class_token_cache = {}
        self.lora_info = {"enabled": False, "wrapped_linears": 0}
        self.pruning_info = {"enabled": False}
        self.denoiser_info = {"enabled": False}

        # ── Frozen Backbones ──
        self.backbone = DualBackbone(config)

        # ── Module 1: Hierarchical Semantic-Visual Synergy ──
        hsvs_cfg = config["hsvs"]
        self.hsvs = HSVS(
            d_clip=hsvs_cfg["d_clip"],
            d_dino=hsvs_cfg["d_dino"],
            d_proj=hsvs_cfg["d_proj"],
            d_head=hsvs_cfg["d_head"],
            num_heads=hsvs_cfg["num_heads"],
            num_layers=hsvs_cfg["num_layers"],
            mlp_ratio=hsvs_cfg["mlp_ratio"],
        )

        # ── Module 2: Vision-Conditioned Prompt Generator ──
        self.vcpg = VCPG(config)

        # ── Module 3: Visual-Text Anomaly Mapper ──
        vtam_cfg = config["vtam"]
        entropy_aware = bool(vtam_cfg.get("entropy_aware", False))
        gamma_default = float(vtam_cfg["gamma"])
        gamma_min_default = 0.15 if entropy_aware else gamma_default
        gamma_max_default = 0.85 if entropy_aware else gamma_default
        self.vtam = VTAM(
            d_proj=hsvs_cfg["d_proj"],
            num_layers=hsvs_cfg["num_layers"],
            tau=vtam_cfg["tau"],
            gamma=vtam_cfg["gamma"],
            entropy_aware=entropy_aware,
            gamma_min=float(vtam_cfg.get("gamma_min", gamma_min_default)),
            gamma_max=float(vtam_cfg.get("gamma_max", gamma_max_default)),
            local_quantile=float(vtam_cfg.get("local_quantile", 0.995)),
            gamma_hidden=int(vtam_cfg.get("gamma_hidden", 16)),
        )

        # Optional denoiser trained on noisy inputs before frozen backbones.
        denoiser_cfg = config.get("denoiser", {})
        denoiser_enabled = bool(denoiser_cfg.get("enabled", False))
        self.denoiser = None
        self.denoiser_noise_cfg = {}
        self.denoiser_info = {
            "enabled": denoiser_enabled,
            "train_on_noisy": bool(denoiser_cfg.get("train_on_noisy", True)),
            "apply_at_inference": bool(denoiser_cfg.get("apply_at_inference", True)),
        }
        if denoiser_enabled:
            base_channels = int(denoiser_cfg.get("base_channels", 24))
            dropout = float(denoiser_cfg.get("dropout", 0.0))
            self.denoiser = DenoisingAutoencoder(base_channels=base_channels, dropout=dropout)
            self.denoiser_noise_cfg = {
                "gaussian_std": float(denoiser_cfg.get("train_noise", {}).get("gaussian_std", 0.10)),
                "dropout_prob": float(denoiser_cfg.get("train_noise", {}).get("dropout_prob", 0.0)),
                "salt_pepper_prob": float(denoiser_cfg.get("train_noise", {}).get("salt_pepper_prob", 0.0)),
                "salt_value": float(denoiser_cfg.get("train_noise", {}).get("salt_value", 2.5)),
                "pepper_value": float(denoiser_cfg.get("train_noise", {}).get("pepper_value", -2.5)),
                "clamp_min": float(denoiser_cfg.get("train_noise", {}).get("clamp_min", -4.0)),
                "clamp_max": float(denoiser_cfg.get("train_noise", {}).get("clamp_max", 4.0)),
            }
            self.denoiser_info.update({
                "base_channels": base_channels,
                "dropout": dropout,
                "noise": self.denoiser_noise_cfg,
            })

        # Optional head-only pruning before optimizer/scheduler creation.
        self._apply_head_pruning(config.get("head_pruning", {}))

        # Segmentation pathway (VTAM/MoE) must always remain fully trainable.
        for p in self.vtam.parameters():
            p.requires_grad = True

        # ── Optional LoRA adapters ──
        lora_cfg = config.get("lora", {})
        lora_scopes = []
        if bool(lora_cfg.get("enabled", False)):
            scopes_raw = lora_cfg.get("scopes", ["backbone"])
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            lora_scopes = [str(s).strip().lower() for s in scopes_raw if str(s).strip()]
            if not lora_scopes:
                lora_scopes = ["backbone"]

            valid_scopes = {"backbone", "hsvs", "vcpg", "vtam"}
            invalid_scopes = [scope for scope in lora_scopes if scope not in valid_scopes]
            if invalid_scopes:
                raise ValueError(
                    "Unsupported LoRA scopes: "
                    f"{invalid_scopes}. Supported scopes: {sorted(valid_scopes)}"
                )

            rank = int(lora_cfg.get("rank", 8))
            alpha = float(lora_cfg.get("alpha", 16.0))
            dropout = float(lora_cfg.get("dropout", 0.0))
            freeze_base = bool(lora_cfg.get("freeze_base", True))

            total_wrapped = 0
            wrapped_by_scope = {}

            # Backbone defaults keep old behavior.
            target_substrings = lora_cfg.get(
                "target_substrings",
                ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            )
            min_block_index = int(lora_cfg.get("min_block_index", 0))

            def _extract_block_index(module_name: str):
                for pat in (r"(?:^|\.)blocks\.(\d+)\.", r"(?:^|\.)resblocks\.(\d+)\."):
                    m = re.search(pat, module_name)
                    if m is not None:
                        return int(m.group(1))
                return None

            def _should_apply_backbone_lora(module_name: str) -> bool:
                lname = module_name.lower()
                block_idx = _extract_block_index(lname)
                if block_idx is not None and block_idx < min_block_index:
                    return False
                if not target_substrings:
                    return True
                return any(tok.lower() in lname for tok in target_substrings)

            if "backbone" in lora_scopes:
                n_clip = apply_lora_to_module(
                    self.backbone.clip_encoder.visual,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                    target_substrings=target_substrings,
                    should_apply=_should_apply_backbone_lora,
                )
                n_dino = apply_lora_to_module(
                    self.backbone.dino_encoder.model,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                    target_substrings=target_substrings,
                    should_apply=_should_apply_backbone_lora,
                )
                wrapped_by_scope["backbone"] = int(n_clip + n_dino)
                total_wrapped += int(n_clip + n_dino)

            head_target_substrings = lora_cfg.get("head_target_substrings", [])

            def _apply_scope(scope_name, module):
                nonlocal total_wrapped
                if scope_name not in lora_scopes:
                    return
                wrapped = apply_lora_to_module(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                    target_substrings=head_target_substrings,
                )
                wrapped_by_scope[scope_name] = int(wrapped)
                total_wrapped += int(wrapped)

            _apply_scope("hsvs", self.hsvs)
            _apply_scope("vcpg", self.vcpg)
            _apply_scope("vtam", self.vtam)

            self.lora_info = {
                "enabled": True,
                "wrapped_linears": int(total_wrapped),
                "wrapped_by_scope": wrapped_by_scope,
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "min_block_index": min_block_index,
                "scopes": lora_scopes,
            }

        # Safety checks: keep VTAM untouched unless explicitly requested.
        vtam_lora_enabled = bool(self.lora_info.get("enabled", False) and "vtam" in lora_scopes)
        if not vtam_lora_enabled:
            for name, module in self.vtam.named_modules():
                if isinstance(module, LoRALinear):
                    raise RuntimeError(f"LoRA must not wrap VTAM module: vtam.{name}")
            if any(not p.requires_grad for p in self.vtam.parameters()):
                raise RuntimeError("All VTAM parameters must remain trainable")

    @staticmethod
    def _validate_keep_ratio(value, field_name):
        keep_ratio = float(value)
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"{field_name} must be in (0, 1], got {keep_ratio}")
        return keep_ratio

    def _head_modules(self):
        return {
            "hsvs": self.hsvs,
            "vcpg": self.vcpg,
            "vtam": self.vtam,
        }

    def _apply_structured_width_pruning(self, module, amount):
        amount = float(amount)
        if amount <= 0.0:
            return 0

        pruned_layers = 0
        for _, child in module.named_modules():
            if not isinstance(child, (nn.Linear, nn.Conv2d)):
                continue
            if hasattr(child, "weight_mask"):
                continue

            out_dim = int(child.weight.shape[0])
            if out_dim <= 1:
                continue

            torch_prune.ln_structured(child, name="weight", amount=amount, n=2, dim=0)
            pruned_layers += 1

        return pruned_layers

    def _apply_depth_pruning(self, keep_ratio):
        keep_ratio = self._validate_keep_ratio(keep_ratio, "head_pruning.depth_keep_ratio")

        total_layers = min(
            int(self.hsvs.num_layers),
            int(self.vtam.moe.num_layers),
            len(self.hsvs.local_atf_blocks),
            len(self.vtam.moe.local_gates),
        )
        if total_layers <= 0:
            raise RuntimeError("Cannot apply depth pruning: no head layers available")

        keep_layers = max(1, min(total_layers, int(round(total_layers * keep_ratio))))
        if keep_layers < total_layers:
            self.hsvs.local_atf_blocks = nn.ModuleList(list(self.hsvs.local_atf_blocks)[:keep_layers])
            self.hsvs.num_layers = keep_layers

            self.vtam.moe.local_gates = nn.ModuleList(list(self.vtam.moe.local_gates)[:keep_layers])
            self.vtam.moe.num_layers = keep_layers

        return {
            "depth_keep_ratio": keep_ratio,
            "depth_total_layers": total_layers,
            "depth_kept_layers": keep_layers,
        }

    def _apply_head_pruning(self, pruning_cfg):
        if not bool(pruning_cfg.get("enabled", False)):
            self.pruning_info = {"enabled": False}
            return

        mode = str(pruning_cfg.get("mode", "width")).strip().lower()
        info = {
            "enabled": True,
            "mode": mode,
        }

        head_modules = self._head_modules()

        if mode == "width":
            keep_ratio = self._validate_keep_ratio(
                pruning_cfg.get("width_keep_ratio", 0.75),
                "head_pruning.width_keep_ratio",
            )
            amount = 1.0 - keep_ratio
            pruned_layers = {
                name: self._apply_structured_width_pruning(module, amount)
                for name, module in head_modules.items()
            }
            info["width_keep_ratio"] = keep_ratio
            info["pruned_layers"] = pruned_layers

        elif mode == "depth":
            info.update(
                self._apply_depth_pruning(
                    pruning_cfg.get("depth_keep_ratio", 0.75)
                )
            )

        elif mode == "differentiated":
            module_keep_ratios = pruning_cfg.get(
                "module_keep_ratios",
                {"hsvs": 0.70, "vcpg": 0.85, "vtam": 0.65},
            )

            normalized_keep_ratios = {}
            pruned_layers = {}
            for name, module in head_modules.items():
                keep_ratio = self._validate_keep_ratio(
                    module_keep_ratios.get(name, 1.0),
                    f"head_pruning.module_keep_ratios.{name}",
                )
                normalized_keep_ratios[name] = keep_ratio
                pruned_layers[name] = self._apply_structured_width_pruning(module, 1.0 - keep_ratio)

            info["module_keep_ratios"] = normalized_keep_ratios
            info["pruned_layers"] = pruned_layers

            if "depth_keep_ratio" in pruning_cfg:
                info.update(self._apply_depth_pruning(pruning_cfg.get("depth_keep_ratio")))

        else:
            raise ValueError(
                f"Unsupported head_pruning.mode '{mode}'. Supported modes: width, depth, differentiated"
            )

        self.pruning_info = info

    def get_trainable_params(self):
        """
        Get trainable parameters grouped by learning rate.

        Returns two groups (§4.2):
            - prompt_params: Prompt embeddings (lr = 5e-4)
            - network_params: VAE, projections, gates (lr = 1e-4)
        """
        prompt_params = []
        network_params = []
        prompt_ids = set()
        network_ids = set()

        def _add_unique(dst, dst_ids, params):
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in dst_ids:
                    continue
                dst.append(p)
                dst_ids.add(pid)

        # VCPG prompt bank → higher LR
        for name, param in self.vcpg.prompt_bank.named_parameters():
            if param.requires_grad:
                _add_unique(prompt_params, prompt_ids, [param])

        # Alpha gating scalar → higher LR (same as prompts)
        if self.vcpg.alpha.requires_grad:
            _add_unique(prompt_params, prompt_ids, [self.vcpg.alpha])

        # Everything else trainable → lower LR
        trainable_modules = [self.hsvs, self.vcpg.vae, self.vcpg.cross_attn,
                             self.vcpg.ln_final, self.vtam]
        if hasattr(self.vcpg, 'text_proj') and not isinstance(self.vcpg.text_proj, nn.Identity):
            trainable_modules.append(self.vcpg.text_proj)
        if self.denoiser is not None:
            trainable_modules.append(self.denoiser)

        for module in trainable_modules:
            _add_unique(network_params, network_ids, module.parameters())

        # Include trainable LoRA adapters from frozen backbones.
        _add_unique(network_params, network_ids, self.backbone.parameters())

        return [
            {"params": prompt_params, "lr_key": "lr_prompt"},
            {"params": network_params, "lr_key": "lr_network"},
        ]

    def count_parameters(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable}

    @torch.no_grad()
    def get_class_token_embedding(self, class_name, device=None):
        """
        Build a semantic class token embedding from frozen CLIP token embeddings.

        Args:
            class_name: category text label (e.g., "cable")
            device: optional device override
        Returns:
            class_emb: [D_text]
        """
        if class_name is None:
            return None

        class_key = str(class_name).strip().lower()
        if not class_key:
            return None

        if device is None:
            device = next(self.parameters()).device

        if class_key in self._class_token_cache:
            return self._class_token_cache[class_key].to(device)

        token_ids = open_clip.tokenize([class_key]).to(device)  # [1, L]
        token_embs = self.backbone.text_encoder.get_token_embedding(token_ids)  # [1, L, D]

        non_pad = torch.nonzero(token_ids[0] != 0, as_tuple=False).flatten()
        if non_pad.numel() > 2:
            content_idx = non_pad[1:-1]  # drop SOT and EOT
            class_emb = token_embs[0, content_idx, :].mean(dim=0)
        elif non_pad.numel() > 1:
            class_emb = token_embs[0, non_pad[1], :]
        else:
            class_emb = token_embs[0, 0, :]

        self._class_token_cache[class_key] = class_emb.detach().cpu()
        return class_emb

    def _build_noisy_inputs(self, images):
        """Apply synthetic corruption to normalized images for denoiser training."""
        cfg = self.denoiser_noise_cfg
        noisy = images

        gaussian_std = float(cfg.get("gaussian_std", 0.0))
        if gaussian_std > 0.0:
            noisy = noisy + torch.randn_like(noisy) * gaussian_std

        dropout_prob = float(cfg.get("dropout_prob", 0.0))
        if dropout_prob > 0.0:
            keep = (
                torch.rand(
                    images.shape[0],
                    1,
                    images.shape[2],
                    images.shape[3],
                    device=images.device,
                )
                >= dropout_prob
            )
            noisy = noisy * keep

        salt_pepper_prob = float(cfg.get("salt_pepper_prob", 0.0))
        if salt_pepper_prob > 0.0:
            rnd = torch.rand(
                images.shape[0],
                1,
                images.shape[2],
                images.shape[3],
                device=images.device,
            )
            salt_mask = rnd < (salt_pepper_prob * 0.5)
            pepper_mask = rnd > (1.0 - salt_pepper_prob * 0.5)
            if bool(salt_mask.any()):
                noisy = torch.where(
                    salt_mask.expand_as(noisy),
                    torch.full_like(noisy, float(cfg.get("salt_value", 2.5))),
                    noisy,
                )
            if bool(pepper_mask.any()):
                noisy = torch.where(
                    pepper_mask.expand_as(noisy),
                    torch.full_like(noisy, float(cfg.get("pepper_value", -2.5))),
                    noisy,
                )

        return torch.clamp(
            noisy,
            min=float(cfg.get("clamp_min", -4.0)),
            max=float(cfg.get("clamp_max", 4.0)),
        )

    def _maybe_denoise_inputs(self, clip_images, dino_images):
        """Optionally denoise noisy train inputs before backbone feature extraction."""
        if self.denoiser is None:
            return clip_images, dino_images, None

        train_on_noisy = bool(self.denoiser_info.get("train_on_noisy", True))
        apply_at_inference = bool(self.denoiser_info.get("apply_at_inference", True))
        shared_source = dino_images is clip_images

        if self.training and train_on_noisy:
            clip_noisy = self._build_noisy_inputs(clip_images)
            dino_noisy = clip_noisy if shared_source else self._build_noisy_inputs(dino_images)
        else:
            clip_noisy = clip_images
            dino_noisy = clip_noisy if shared_source else dino_images

        denoised_clip = self.denoiser(clip_noisy)
        denoised_dino = denoised_clip if shared_source else self.denoiser(dino_noisy)

        denoise_recon_loss = None
        if self.training:
            loss_clip = F.l1_loss(denoised_clip, clip_images)
            if shared_source:
                denoise_recon_loss = loss_clip
            else:
                denoise_recon_loss = 0.5 * (loss_clip + F.l1_loss(denoised_dino, dino_images))

        if self.training or apply_at_inference:
            return denoised_clip, denoised_dino, denoise_recon_loss

        return clip_images, dino_images, denoise_recon_loss

    def forward(self, clip_images, dino_images=None, class_token_embedding=None):
        """
        Full SSVP forward inference pipeline — Algorithm 1.

        Args:
            clip_images:           [B, 3, 518, 518] — input images for CLIP
            dino_images:           [B, 3, 518, 518] — input images for DINOv2
                                   (if None, uses clip_images)
            class_token_embedding: [D_text] — optional class name token embedding

        Returns:
            dict containing:
                anomaly_map:       [B, 1, H, W] — pixel-level anomaly probability
                anomaly_score:     [B] — image-level anomaly score
                mu, logvar:        VAE latent distribution params
                v_syn_global:      [B, D_proj] — global synergistic feature
                v_recon:           [B, D_proj] — VAE reconstructed feature
                t_final_normal:    Vision-conditioned normal prompts
                t_final_abnormal:  Vision-conditioned abnormal prompts
                t_init_normal:     Initial normal prompts
                t_init_abnormal:   Initial abnormal prompts
        """
        if dino_images is None:
            dino_images = clip_images

        clip_images, dino_images, denoise_recon_loss = self._maybe_denoise_inputs(
            clip_images,
            dino_images,
        )

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  Stage 1: Feature Extraction                                   ║
        # ╚══════════════════════════════════════════════════════════════════╝
        if self.lora_info.get("enabled", False):
            # Keep backbone base weights frozen via requires_grad=False,
            # but allow autograd for LoRA adapters.
            clip_global, clip_locals, dino_global, dino_locals = \
                self.backbone(clip_images, dino_images)
        else:
            with torch.no_grad():
                clip_global, clip_locals, dino_global, dino_locals = \
                    self.backbone(clip_images, dino_images)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  Stage 2: Hierarchical Semantic-Visual Synergy (HSVS)          ║
        # ╚══════════════════════════════════════════════════════════════════╝
        v_syn_global, v_syn_locals = self.hsvs(
            clip_global, clip_locals, dino_global, dino_locals
        )
        # v_syn_global: [B, D_proj]
        # v_syn_locals: List of [B, N, D_proj] × num_layers

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  Stage 3: Vision-Conditioned Prompt Generator (VCPG)           ║
        # ╚══════════════════════════════════════════════════════════════════╝
        (t_final_normal, t_final_abnormal,
         t_init_normal, t_init_abnormal,
         vae_outputs) = self.vcpg(v_syn_global, class_token_embedding)

        # Aggregate prompts for VTAM
        t_norm_agg, t_abn_agg = self.vcpg.get_aggregated_prompt_features(
            t_final_normal, t_final_abnormal
        )  # [B, d_text] each

        # Align prompt features to synergistic space if dimensions differ
        d_proj = v_syn_global.shape[-1]
        d_text = t_norm_agg.shape[-1]
        if d_proj != d_text:
            # Use the VCPG's text projection (or its inverse idea)
            t_norm_agg = self.vcpg.text_proj(t_norm_agg)
            t_abn_agg = self.vcpg.text_proj(t_abn_agg)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  Stage 4: Visual-Text Anomaly Mapper (VTAM)                    ║
        # ╚══════════════════════════════════════════════════════════════════╝
        anomaly_map, anomaly_score = self.vtam(
            v_syn_locals, v_syn_global, t_norm_agg, t_abn_agg
        )
        # anomaly_map: [B, 1, H, W]
        # anomaly_score: [B]

        return {
            "anomaly_map": anomaly_map,
            "anomaly_score": anomaly_score,
            "mu": vae_outputs["mu"],
            "logvar": vae_outputs["logvar"],
            "v_syn_global": v_syn_global,
            "v_recon": vae_outputs["v_recon"],
            "t_final_normal": t_final_normal,
            "t_final_abnormal": t_final_abnormal,
            "t_init_normal": t_init_normal,
            "t_init_abnormal": t_init_abnormal,
            "denoise_recon_loss": denoise_recon_loss,
        }
