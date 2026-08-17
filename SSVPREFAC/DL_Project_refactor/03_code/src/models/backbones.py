"""
Dual Backbone Feature Extractor — CLIP ViT-L/14 + DINOv2 ViT-L/14 (fallback for DINOv3).

Both backbones are completely frozen. Multi-scale intermediate features are extracted
via forward hooks for use by the HSVS fusion module.

Resolution Adaptation Strategy (§4.2):
    Since both CLIP ViT-L/14 and DINOv2 ViT-L/14 use patch_size=14,
    we set input_size = 518 for both → 518/14 = 37 grid → [B, 1369, D].
"""

import torch
import torch.nn as nn
import open_clip
import warnings
from pathlib import Path


class CLIPFeatureExtractor(nn.Module):
    """
    Frozen CLIP ViT-L/14 image encoder with multi-scale feature hooks.
    
    Extracts:
        - Global CLS token:  [B, D_clip]
        - Multi-scale patch tokens: List of [B, N, D_clip] at specified layers
    """

    def __init__(self, model_name="ViT-L-14", pretrained="openai",
                 feature_layers=(6, 12, 18, 24), input_size=518):
        super().__init__()
        self.feature_layers = list(feature_layers)
        self.input_size = input_size

        # Load the full CLIP model, then extract image encoder
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        self.embed_dim = self.visual.output_dim  # typically 768 for projection

        # Ensure positional embeddings match the desired input grid size.
        # OpenCLIP can store positional embeddings as either [1+N, D] or [1, 1+N, D].
        pos_embed = getattr(self.visual, "positional_embedding", None)
        if pos_embed is not None:
            pe = pos_embed
            is_2d = pe.dim() == 2
            if is_2d:
                pe = pe.unsqueeze(0)  # [1, 1+N, D]

            seq_len = pe.shape[1]
            cur_grid = int((seq_len - 1) ** 0.5)
            patch_size = 14
            target_grid = self.input_size // patch_size

            if cur_grid * cur_grid + 1 == seq_len and cur_grid != target_grid:
                cls_token = pe[:, :1, :]
                grid_tokens = pe[:, 1:, :].reshape(1, cur_grid, cur_grid, -1).permute(0, 3, 1, 2)
                grid_tokens_resized = torch.nn.functional.interpolate(
                    grid_tokens,
                    size=(target_grid, target_grid),
                    mode="bicubic",
                    align_corners=False,
                )
                grid_tokens_resized = grid_tokens_resized.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, -1)
                new_pos = torch.cat([cls_token, grid_tokens_resized], dim=1)

                if is_2d:
                    new_pos = new_pos.squeeze(0)  # [1+N, D]

                # Preserve parameter type for downstream modules.
                self.visual.positional_embedding = nn.Parameter(new_pos.to(pos_embed.dtype), requires_grad=False)

        # Freeze all parameters
        for param in self.visual.parameters():
            param.requires_grad = False

        # Register hooks on transformer blocks to capture intermediate features
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward hooks to specified transformer layers."""
        transformer = self.visual.transformer
        for idx in self.feature_layers:
            # OpenCLIP layers are 0-indexed; layer indices in config are 1-indexed block counts
            # The resblocks are 0-indexed, so layer 6 means resblocks[5]
            layer_idx = idx - 1  # Convert to 0-indexed
            if layer_idx < len(transformer.resblocks):
                transformer.resblocks[layer_idx].register_forward_hook(
                    self._make_hook(idx)
                )

    def _make_hook(self, layer_id):
        def hook(module, input, output):
            self._features[layer_id] = output
        return hook

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images (pre-resized to self.input_size)
        Returns:
            global_feat:  [B, D_hidden]  CLS token from final layer
            local_feats:  list of [B, N, D_hidden]  patch tokens at each hooked layer
        """
        self._features = {}

        # CLIP visual forward — output format varies across OpenCLIP versions.
        x = self.visual(x)

        def _to_bsd(tensor):
            """Convert token tensor to [B, S, D] if needed."""
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
                return tensor
            # Common formats: [S, B, D] or [B, S, D]
            if tensor.shape[0] > tensor.shape[1]:
                return tensor.permute(1, 0, 2)
            return tensor

        # Collect hooked features
        local_feats = []
        for layer_id in self.feature_layers:
            feat = self._features.get(layer_id, None)
            if feat is not None:
                if isinstance(feat, (tuple, list)) and len(feat) > 0:
                    feat = feat[0]
                feat = _to_bsd(feat)
                # OpenCLIP ViT outputs shape [seq_len, batch, dim] from resblocks
                # feat is [B, 1+N, D] where token 0 is CLS
                local_feats.append(feat[:, 1:, :])  # Remove CLS, keep patches
            else:
                local_feats.append(None)

        # Global feature: CLS token from the final hooked layer
        final_feat = self._features.get(self.feature_layers[-1], None)
        if final_feat is not None:
            if isinstance(final_feat, (tuple, list)) and len(final_feat) > 0:
                final_feat = final_feat[0]
            final_feat = _to_bsd(final_feat)
            global_feat = final_feat[:, 0, :]  # CLS token
        else:
            # Fallback: derive a global feature from the visual output.
            if isinstance(x, (tuple, list)) and len(x) > 0:
                x = x[0]
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                x = _to_bsd(x)
                global_feat = x[:, 0, :]
            else:
                global_feat = x

        return global_feat, local_feats


class DINOFeatureExtractor(nn.Module):
    """
    Frozen DINOv2 ViT-L/14 encoder with multi-scale feature hooks.
    Serves as fallback for DINOv3 (ViT-L/16).

    Extracts:
        - Global CLS token:  [B, D_dino]
        - Multi-scale patch tokens: List of [B, N, D_dino] at specified layers
    """

    def __init__(self, model_name="dinov2_vitl14", feature_layers=(6, 12, 18, 23),
                 input_size=518):
        super().__init__()
        self.feature_layers = list(feature_layers)
        self.input_size = input_size

        # Load DINOv2 from torch hub, with local-cache fallback for offline runs.
        self.model = self._load_dino_model(model_name)
        self.embed_dim = self.model.embed_dim  # 1024 for ViT-L

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks
        self._features = {}
        self._register_hooks()

    @staticmethod
    def _load_dino_model(model_name):
        try:
            return torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:
            hub_cache_dir = Path(torch.hub.get_dir())
            local_repo = hub_cache_dir / "facebookresearch_dinov2_main"
            if local_repo.exists():
                warnings.warn(
                    "Falling back to local DINOv2 hub cache because remote load failed: "
                    f"{exc}"
                )
                return torch.hub.load(str(local_repo), model_name, source="local")
            raise

    def _register_hooks(self):
        """Attach forward hooks to DINOv2 transformer blocks."""
        for idx in self.feature_layers:
            layer_idx = idx - 1  # 0-indexed
            if layer_idx < len(self.model.blocks):
                self.model.blocks[layer_idx].register_forward_hook(
                    self._make_hook(idx)
                )

    def _make_hook(self, layer_id):
        def hook(module, input, output):
            self._features[layer_id] = output
        return hook

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images (pre-resized to self.input_size)
        Returns:
            global_feat:  [B, D_dino]  CLS token
            local_feats:  list of [B, N, D_dino]  patch tokens at each hooked layer
        """
        self._features = {}

        # DINOv2 forward — outputs [B, D] class token by default
        _ = self.model(x)

        # Collect hooked features
        local_feats = []
        for layer_id in self.feature_layers:
            feat = self._features.get(layer_id, None)
            if feat is not None:
                # DINOv2 blocks output [B, 1+N, D]
                local_feats.append(feat[:, 1:, :])  # Remove CLS
            else:
                local_feats.append(None)

        # Global CLS from final hooked layer
        final_feat = self._features.get(self.feature_layers[-1], None)
        if final_feat is not None:
            global_feat = final_feat[:, 0, :]  # CLS token
        else:
            global_feat = self.model(x)

        return global_feat, local_feats


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder. Used to encode learnable prompt embeddings
    into the shared text-visual embedding space.
    """

    def __init__(self, model_name="ViT-L-14", pretrained="openai", int8_dynamic_quant=False):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.attn_mask = model.attn_mask if hasattr(model, "attn_mask") else None

        # Get context length and vocab size
        self.context_length = model.context_length
        self.vocab_size = model.vocab_size
        self.int8_dynamic_quant = False

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        if bool(int8_dynamic_quant):
            if torch.cuda.is_available():
                warnings.warn(
                    "Skipping CLIP text INT8 dynamic quantization because CUDA is active; "
                    "dynamic quantized Linear layers are CPU-only."
                )
            else:
                try:
                    import torch.ao.quantization as tq

                    self.transformer = tq.quantize_dynamic(
                        self.transformer,
                        {nn.Linear},
                        dtype=torch.qint8,
                    )
                    self.int8_dynamic_quant = True
                except Exception as exc:
                    warnings.warn(f"Failed to apply CLIP text INT8 dynamic quantization: {exc}")

    @torch.no_grad()
    def encode_text_embeddings(self, text_embeddings, tokenized_text=None):
        """
        Encode pre-built prompt embeddings through the frozen text transformer.

        Args:
            text_embeddings: [B, seq_len, D_text] — full prompt embedding sequences
                            (already includes positional info or will be added)
            tokenized_text: Optional [B, seq_len] token IDs for EOT position detection

        Returns:
            text_features: [B, D_text] — text features (taken at EOT token position)
        """
        x = text_embeddings

        # Add positional embedding if dimensions match
        if self.positional_embedding is not None:
            seq_len = x.shape[1]
            x = x + self.positional_embedding[:seq_len].to(x.dtype)

        # Pass through transformer
        # OpenCLIP expects [seq_len, batch, dim]
        x = x.permute(1, 0, 2)

        # Build causal mask
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]].to(x.device, x.dtype)
        else:
            attn_mask = None

        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # [B, seq_len, D]
        x = self.ln_final(x)

        # Take features from the EOT token position
        # If tokenized_text provided, find EOT; otherwise use last token
        if tokenized_text is not None:
            # EOT is the argmax of token IDs (highest token ID is EOT in CLIP)
            eot_pos = tokenized_text.argmax(dim=-1)
            x = x[torch.arange(x.shape[0], device=x.device), eot_pos]
        else:
            x = x[:, -1, :]  # Last token as default

        # Project to shared embedding space
        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.no_grad()
    def get_token_embedding(self, token_ids):
        """Get token embeddings from vocabulary IDs."""
        return self.token_embedding(token_ids)


class DualBackbone(nn.Module):
    """
    Combined frozen backbone: CLIP Image + Text Encoder + DINOv2.

    All encoders are frozen — no gradients flow through them.
    This module provides the raw multi-scale features used by downstream
    trainable modules (HSVS, VCPG, VTAM).
    """

    def __init__(self, config):
        super().__init__()
        bb_cfg = config["backbone"]

        self.clip_encoder = CLIPFeatureExtractor(
            model_name=bb_cfg["clip_model"],
            pretrained=bb_cfg["clip_pretrained"],
            feature_layers=bb_cfg["clip_feature_layers"],
            input_size=bb_cfg["clip_input_size"],
        )
        self.dino_encoder = DINOFeatureExtractor(
            model_name=bb_cfg["dino_model"],
            feature_layers=bb_cfg["dino_feature_layers"],
            input_size=bb_cfg.get("dino_input_size", bb_cfg["clip_input_size"]),
        )
        self.text_encoder = CLIPTextEncoder(
            model_name=bb_cfg["clip_model"],
            pretrained=bb_cfg["clip_pretrained"],
            int8_dynamic_quant=bool(bb_cfg.get("text_int8_dynamic_quant", False)),
        )

        self.d_clip = bb_cfg["d_clip"]
        self.d_dino = bb_cfg["d_dino"]

    def extract_visual_features(self, clip_images, dino_images=None):
        """
        Extract multi-scale visual features from both backbones.

        Args:
            clip_images:  [B, 3, 518, 518]
            dino_images:  [B, 3, 518, 518] (same size since DINOv2 uses patch=14)

        Returns:
            clip_global:   [B, D_clip]
            clip_locals:   List of 4× [B, N, D_clip]
            dino_global:   [B, D_dino]
            dino_locals:   List of 4× [B, N, D_dino]

        Note:
            Autograd behavior is controlled by the caller (`SSVP.forward`).
            This allows LoRA adapters on backbone linears to receive gradients
            while keeping frozen-backbone inference paths under `torch.no_grad()`.
        """
        if dino_images is None:
            dino_images = clip_images

        clip_global, clip_locals = self.clip_encoder(clip_images)
        dino_global, dino_locals = self.dino_encoder(dino_images)

        return clip_global, clip_locals, dino_global, dino_locals

    def forward(self, clip_images, dino_images=None):
        return self.extract_visual_features(clip_images, dino_images)
