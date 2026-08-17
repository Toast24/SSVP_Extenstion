"""
Hierarchical Semantic-Visual Synergy (HSVS) Module — §3.1

Bridges the gap between CLIP's global semantics and DINOv2's fine-grained structural
features using Adaptive Token Features Fusion (ATF) blocks with dual-path cross-modal
attention.

Architecture:
    For each scale l ∈ {1..4}:
        1. Project CLIP & DINO tokens to shared subspace via W_Q, W_K, W_V (Eq. 2)
        2. Compute dual cross-attention: Attn_c→d and Attn_d→c (Eq. 3)
        3. Concatenate + LN → MLP fusion → residual connection (Eq. 4-5)
        Output: V_syn^local,l = F_l^c + MLP([LN(Attn_c→d) ‖ LN(Attn_d→c)])

    Same ATF applied to global CLS tokens → V_syn^global
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ATFBlock(nn.Module):
    """
    Adaptive Token Features Fusion (ATF) Block — single scale.

    Implements dual-path cross-modal attention between CLIP and DINO features
    in a shared latent subspace, followed by MLP fusion with residual connection.

    Args:
        d_clip:    CLIP feature dimension (1024 for ViT-L)
        d_dino:    DINO feature dimension (1024 for ViT-L)
        d_head:    Attention head dimension (128)
        num_heads: Number of attention heads (6)
        d_proj:    Output projection dimension (768)
        mlp_ratio: MLP expansion ratio (4)
    """

    def __init__(self, d_clip=1024, d_dino=1024, d_head=128, num_heads=6,
                 d_proj=768, mlp_ratio=4):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        self.inner_dim = d_head * num_heads  # Total attention dim
        self.scale = math.sqrt(d_head)

        # ── Path 1: CLIP queries DINO (Attn_c→d) ──
        # Q from CLIP, K/V from DINO
        self.W_Q_c = nn.Linear(d_clip, self.inner_dim, bias=False)
        self.W_K_d = nn.Linear(d_dino, self.inner_dim, bias=False)
        self.W_V_d = nn.Linear(d_dino, self.inner_dim, bias=False)

        # ── Path 2: DINO queries CLIP (Attn_d→c) ──
        # Q from DINO, K/V from CLIP
        self.W_Q_d = nn.Linear(d_dino, self.inner_dim, bias=False)
        self.W_K_c = nn.Linear(d_clip, self.inner_dim, bias=False)
        self.W_V_c = nn.Linear(d_clip, self.inner_dim, bias=False)

        # ── Output projections for each attention path ──
        self.out_proj_cd = nn.Linear(self.inner_dim, d_proj)
        self.out_proj_dc = nn.Linear(self.inner_dim, d_proj)

        # ── Layer norms for each path (Eq. 4) ──
        self.ln_cd = nn.LayerNorm(d_proj)
        self.ln_dc = nn.LayerNorm(d_proj)

        # ── MLP fusion: 2-layer with GELU (processes concatenated features) ──
        # Input: concatenation of both paths → 2 * d_proj
        mlp_hidden = d_proj * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_proj, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, d_proj),
            nn.Dropout(0.1),
        )

        # ── Residual projection: align CLIP dim to output dim if different ──
        if d_clip != d_proj:
            self.residual_proj = nn.Linear(d_clip, d_proj, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def _multi_head_attention(self, Q, K, V):
        """
        Standard multi-head scaled dot-product attention.

        Args:
            Q: [B, N, inner_dim]
            K: [B, M, inner_dim]
            V: [B, M, inner_dim]
        Returns:
            out: [B, N, inner_dim]
        """
        B, N, _ = Q.shape
        M = K.shape[1]

        # Reshape for multi-head: [B, num_heads, seq_len, d_head]
        Q = Q.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, M, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, M, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N, M]
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, V)  # [B, H, N, d_head]

        # Reshape back: [B, N, inner_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        return out

    def forward(self, feat_clip, feat_dino):
        """
        Adaptive Token Features Fusion at a single scale.

        Args:
            feat_clip:  [B, N, D_clip]  — CLIP patch tokens (or global CLS)
            feat_dino:  [B, N, D_dino]  — DINO patch tokens (or global CLS)

        Returns:
            v_syn: [B, N, D_proj]  — Synergistic feature at this scale
        """
        # ── Path 1: CLIP queries DINO → injects structural priors into semantics ──
        Q_c = self.W_Q_c(feat_clip)   # [B, N, inner_dim]
        K_d = self.W_K_d(feat_dino)   # [B, N, inner_dim]
        V_d = self.W_V_d(feat_dino)   # [B, N, inner_dim]
        attn_cd = self._multi_head_attention(Q_c, K_d, V_d)  # [B, N, inner_dim]
        attn_cd = self.out_proj_cd(attn_cd)  # [B, N, d_proj]

        # ── Path 2: DINO queries CLIP → structural reinforcement ──
        Q_d = self.W_Q_d(feat_dino)   # [B, N, inner_dim]
        K_c = self.W_K_c(feat_clip)   # [B, N, inner_dim]
        V_c = self.W_V_c(feat_clip)   # [B, N, inner_dim]
        attn_dc = self._multi_head_attention(Q_d, K_c, V_c)  # [B, N, inner_dim]
        attn_dc = self.out_proj_dc(attn_dc)  # [B, N, d_proj]

        # ── Concatenate with LayerNorm (Eq. 4) ──
        z_joint = torch.cat([self.ln_cd(attn_cd), self.ln_dc(attn_dc)], dim=-1)
        # z_joint: [B, N, 2*d_proj]

        # ── MLP fusion + residual connection (Eq. 5) ──
        fused = self.mlp(z_joint)  # [B, N, d_proj]
        residual = self.residual_proj(feat_clip)  # [B, N, d_proj]
        v_syn = residual + fused

        return v_syn


class HSVS(nn.Module):
    """
    Hierarchical Semantic-Visual Synergy (HSVS) — §3.1

    Applies ATF blocks across multiple scales (4 local + 1 global) to produce
    synergistic features that combine CLIP semantics with DINO structures.

    Args:
        d_clip:     CLIP hidden dim (1024)
        d_dino:     DINO hidden dim (1024)
        d_proj:     Output synergistic dim (768)
        d_head:     Attention head dim (128)
        num_heads:  Number of attention heads (6)
        num_layers: Number of scale levels (4)
        mlp_ratio:  MLP expansion ratio (4)
    """

    def __init__(self, d_clip=1024, d_dino=1024, d_proj=768, d_head=128,
                 num_heads=6, num_layers=4, mlp_ratio=4):
        super().__init__()
        self.num_layers = num_layers

        # Local ATF blocks — one per scale
        self.local_atf_blocks = nn.ModuleList([
            ATFBlock(d_clip, d_dino, d_head, num_heads, d_proj, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Global ATF block — for CLS tokens
        self.global_atf_block = ATFBlock(
            d_clip, d_dino, d_head, num_heads, d_proj, mlp_ratio
        )

    def forward(self, clip_global, clip_locals, dino_global, dino_locals):
        """
        Produce synergistic features at all scales.

        Args:
            clip_global:  [B, D_clip]                — CLIP CLS token
            clip_locals:  List of [B, N, D_clip] ×4  — CLIP patch tokens per layer
            dino_global:  [B, D_dino]                — DINO CLS token
            dino_locals:  List of [B, N, D_dino] ×4  — DINO patch tokens per layer

        Returns:
            v_syn_global: [B, D_proj]               — Global synergistic feature
            v_syn_locals: List of [B, N, D_proj] ×4 — Multi-scale local features
        """
        # ── Local synergistic features (multi-scale) ──
        v_syn_locals = []
        for l in range(self.num_layers):
            clip_l = clip_locals[l]  # [B, N, D_clip]
            dino_l = dino_locals[l]  # [B, N, D_dino]

            if clip_l is None or dino_l is None:
                v_syn_locals.append(None)
                continue

            # Handle potential spatial dimension mismatch via interpolation
            if clip_l.shape[1] != dino_l.shape[1]:
                # Reshape to spatial, interpolate, reshape back
                B, N_c, D_c = clip_l.shape
                N_d = dino_l.shape[1]
                H_c = int(math.sqrt(N_c))
                H_d = int(math.sqrt(N_d))

                dino_l = dino_l.transpose(1, 2).view(B, -1, H_d, H_d)
                dino_l = F.interpolate(dino_l, size=(H_c, H_c), mode="bilinear",
                                       align_corners=False)
                dino_l = dino_l.view(B, -1, N_c).transpose(1, 2)

            v_syn_l = self.local_atf_blocks[l](clip_l, dino_l)
            v_syn_locals.append(v_syn_l)

        # ── Global synergistic feature ──
        # Expand CLS tokens to [B, 1, D] for ATF compatibility
        clip_g = clip_global.unsqueeze(1)   # [B, 1, D_clip]
        dino_g = dino_global.unsqueeze(1)   # [B, 1, D_dino]
        v_syn_global = self.global_atf_block(clip_g, dino_g)  # [B, 1, D_proj]
        v_syn_global = v_syn_global.squeeze(1)  # [B, D_proj]

        return v_syn_global, v_syn_locals
