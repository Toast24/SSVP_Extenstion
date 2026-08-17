"""
Visual-Text Anomaly Mapper (VTAM) — §3.3

Resolves the global-local scoring disconnect using an Anomaly Mixture-of-Experts
(AnomalyMoE) with dual-gating mechanism.

Pipeline:
    1. Compute per-layer anomaly probability maps via cosine similarity (Eq. 13-14)
    2. Global Scale Gating: MLP on V_syn^global → softmax scale weights (Eq. 15)
    3. Local Spatial Gating: 1×1 conv on V_syn^local → sigmoid attention mask (Eq. 16)
    4. Dual-Gated Aggregation: P_map = Σ w_scale[l] · (M_spatial^l ⊙ P_anom^l) (Eq. 17)
    5. Score Enhancement: S_final = (1-γ)·S_global + γ·S_local (Eq. 18-19)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AnomalyMoE(nn.Module):
    """
    Anomaly Mixture-of-Experts — §3.3.2, Algorithm 2

    Dual-gated aggregation of multi-scale anomaly experts:
        - Branch A: Global Scale Gating (inter-layer weighting)
        - Branch B: Local Spatial Gating (intra-layer attention)

    Args:
        d_proj:     Synergistic feature dim (768)
        num_layers: Number of scale levels / experts (4)
        tau:        Temperature for anomaly probability softmax (0.07)
    """

    def __init__(self, d_proj=768, num_layers=4, tau=0.07, entropy_aware=True):
        super().__init__()
        self.num_layers = num_layers
        self.tau = nn.Parameter(torch.tensor(tau))
        self.entropy_aware = bool(entropy_aware)

        # Branch A: Global Scale Gating — F_gate^global (Eq. 15)
        # MLP that maps V_syn^global to per-layer importance weights
        self.global_gate = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj // 2, num_layers),
        )

        # Branch B: Local Spatial Gating — F_gate^local per layer (Eq. 16)
        # 1×1 convolution producing spatial attention mask
        self.local_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_proj, d_proj // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(d_proj // 4, 1, kernel_size=1),
            )
            for _ in range(num_layers)
        ])

    @staticmethod
    def _mean_binary_entropy(prob_map):
        p = prob_map.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        # Normalize to [0, 1] by dividing by ln(2), then average spatially.
        return (entropy / math.log(2.0)).mean(dim=(1, 2, 3))

    def compute_anomaly_probability(self, v_local, t_normal, t_abnormal, tau):
        """
        Compute pixel-wise anomaly probability map for a single layer — Eq. 13-14.

        Args:
            v_local:    [B, N, D] — synergistic local features (patch tokens)
            t_normal:   [B, D]   — aggregated normal prompt embedding
            t_abnormal: [B, D]   — aggregated abnormal prompt embedding
            tau:        Temperature scalar

        Returns:
            p_anom: [B, 1, H, W] — anomaly probability map ∈ [0, 1]
        """
        B, N, D = v_local.shape
        H = W = int(math.sqrt(N))

        # Normalize features for cosine similarity
        v_norm = F.normalize(v_local, dim=-1)             # [B, N, D]
        t_norm_n = F.normalize(t_normal, dim=-1)          # [B, D]
        t_norm_a = F.normalize(t_abnormal, dim=-1)        # [B, D]

        # Dense cosine similarity (Eq. 13)
        sim_normal = torch.einsum("bnd,bd->bn", v_norm, t_norm_n)    # [B, N]
        sim_abnormal = torch.einsum("bnd,bd->bn", v_norm, t_norm_a)  # [B, N]

        # Stack and apply softmax with temperature (Eq. 14)
        sim_stack = torch.stack([sim_normal, sim_abnormal], dim=-1)  # [B, N, 2]
        probs = F.softmax(sim_stack / tau, dim=-1)                   # [B, N, 2]

        # Extract anomaly channel probability
        p_anom = probs[:, :, 1]  # [B, N]
        p_anom = p_anom.view(B, 1, H, W)  # [B, 1, H, W]

        return p_anom

    def forward(self, v_syn_locals, v_syn_global, t_normal, t_abnormal):
        """
        AnomalyMoE forward — Algorithm 2.

        Args:
            v_syn_locals: List of [B, N, D] ×num_layers — multi-scale local features
            v_syn_global: [B, D] — global synergistic feature
            t_normal:     [B, D] — aggregated normal prompt
            t_abnormal:   [B, D] — aggregated abnormal prompt

        Returns:
            p_map: [B, 1, H, W] — fused anomaly probability map
        """
        B = v_syn_global.shape[0]
        tau = self.tau.abs() + 1e-4  # Ensure positive temperature

        # ── Branch A: Global Scale Gating (Eq. 15) ──
        w_scale = F.softmax(self.global_gate(v_syn_global), dim=-1)  # [B, num_layers]

        # ── Generate raw expert maps + Branch B: Local Spatial Gating ──
        fused_layers = []
        layer_confidences = []
        active_indices = []

        for l in range(self.num_layers):
            v_local = v_syn_locals[l]
            if v_local is None:
                continue

            B, N, D = v_local.shape
            H = W = int(math.sqrt(N))

            # Step 1: Raw anomaly probability (Eq. 13-14)
            p_anom_l = self.compute_anomaly_probability(
                v_local, t_normal, t_abnormal, tau
            )  # [B, 1, H, W]

            # Step 2: Local spatial gating (Eq. 16)
            v_spatial = v_local.transpose(1, 2).view(B, D, H, W)  # [B, D, H, W]
            m_spatial = torch.sigmoid(self.local_gates[l](v_spatial))  # [B, 1, H, W]

            # Step 3: Dual-gated contribution (Eq. 17)
            fused_l = m_spatial * p_anom_l  # [B, 1, H, W]
            fused_layers.append(fused_l)
            active_indices.append(l)

            if self.entropy_aware:
                # Low entropy map => high confidence for this expert.
                entropy_l = self._mean_binary_entropy(p_anom_l)
                conf_l = (1.0 - entropy_l).clamp(0.05, 1.0)
            else:
                conf_l = torch.ones(B, device=v_syn_global.device, dtype=v_syn_global.dtype)
            layer_confidences.append(conf_l)

        if len(fused_layers) == 0:
            raise RuntimeError("AnomalyMoE received no valid local features.")

        # Entropy-aware reweighting over active experts.
        base_weights = w_scale[:, active_indices]  # [B, L_active]
        conf_stack = torch.stack(layer_confidences, dim=1)  # [B, L_active]
        adaptive_weights = base_weights * conf_stack
        adaptive_weights = adaptive_weights / adaptive_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        p_map = None
        for idx, fused_l in enumerate(fused_layers):
            w_l = adaptive_weights[:, idx].view(B, 1, 1, 1)
            p_weighted = w_l * fused_l
            if p_map is None:
                p_map = p_weighted
            else:
                if p_weighted.shape[-2:] != p_map.shape[-2:]:
                    p_weighted = F.interpolate(
                        p_weighted,
                        size=p_map.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                p_map = p_map + p_weighted

        return p_map


class VTAM(nn.Module):
    """
    Visual-Text Anomaly Mapper — §3.3

    Full pipeline: AnomalyMoE → anomaly map → score enhancement.

    Args:
        d_proj:     Feature dimension (768)
        num_layers: Number of scale levels (4)
        tau:        Temperature (0.07)
        gamma:      Score balance factor (0.5) — Eq. 19
    """

    def __init__(
        self,
        d_proj=768,
        num_layers=4,
        tau=0.07,
        gamma=0.5,
        entropy_aware=True,
        gamma_min=0.15,
        gamma_max=0.85,
        local_quantile=0.995,
        gamma_hidden=16,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.entropy_aware = bool(entropy_aware)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.local_quantile = float(local_quantile)

        self.gamma_predictor = None
        if self.entropy_aware:
            self.gamma_predictor = nn.Sequential(
                nn.Linear(3, int(gamma_hidden)),
                nn.GELU(),
                nn.Linear(int(gamma_hidden), 1),
            )

        self.moe = AnomalyMoE(
            d_proj=d_proj,
            num_layers=num_layers,
            tau=tau,
            entropy_aware=entropy_aware,
        )

    @staticmethod
    def _mean_binary_entropy(prob_map):
        p = prob_map.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return (entropy / math.log(2.0)).mean(dim=(1, 2, 3))

    def _local_score(self, p_map):
        flat = p_map.view(p_map.shape[0], -1)
        q_score = torch.quantile(flat, q=self.local_quantile, dim=-1)
        mean_score = flat.mean(dim=-1)
        return 0.7 * q_score + 0.3 * mean_score

    def compute_anomaly_map(self, v_syn_locals, v_syn_global, t_normal, t_abnormal):
        """
        Compute the fused pixel-level anomaly map via AnomalyMoE.

        Args:
            v_syn_locals: List of [B, N, D] ×4
            v_syn_global: [B, D]
            t_normal:     [B, D]
            t_abnormal:   [B, D]

        Returns:
            p_map: [B, 1, H, W] — anomaly probability map
        """
        return self.moe(v_syn_locals, v_syn_global, t_normal, t_abnormal)

    def compute_score(self, v_syn_global, t_normal, t_abnormal, p_map):
        """
        Score Enhancement Strategy — §3.3.3, Eq. 18-19.

        Combines global semantic score with local anomaly evidence.

        Args:
            v_syn_global: [B, D]
            t_normal:     [B, D]
            t_abnormal:   [B, D]
            p_map:        [B, 1, H, W]

        Returns:
            s_final: [B] — final anomaly score
        """
        # Global score: cosine similarity difference (abnormal - normal)
        v_norm = F.normalize(v_syn_global, dim=-1)
        t_n_norm = F.normalize(t_normal, dim=-1)
        t_a_norm = F.normalize(t_abnormal, dim=-1)

        s_global_abn = torch.sum(v_norm * t_a_norm, dim=-1)  # [B]
        s_global_norm = torch.sum(v_norm * t_n_norm, dim=-1)  # [B]
        s_global = s_global_abn - s_global_norm  # Higher = more anomalous

        # Local evidence with entropy-aware robust pooling.
        p_prob = p_map.clamp(1e-6, 1.0 - 1e-6)
        s_local = self._local_score(p_prob)  # [B]

        # Dynamic fusion coefficient from entropy + confidence cues.
        entropy_norm = self._mean_binary_entropy(p_prob)  # [B], lower is better
        local_conf = (1.0 - entropy_norm).clamp(0.0, 1.0)
        semantic_conf = torch.sigmoid(s_global)

        if self.entropy_aware and self.gamma_predictor is not None:
            gate_features = torch.stack([entropy_norm, local_conf, semantic_conf], dim=-1)
            gamma_dyn = torch.sigmoid(self.gamma_predictor(gate_features)).squeeze(-1)
            gamma_dyn = 0.5 * gamma_dyn + 0.5 * self.gamma
        else:
            gamma_dyn = torch.full_like(s_global, self.gamma)

        gamma_dyn = gamma_dyn.clamp(self.gamma_min, self.gamma_max)

        # Final score (entropy-aware Eq. 19 variant)
        s_final = (1.0 - gamma_dyn) * s_global + gamma_dyn * s_local

        return s_final

    def forward(self, v_syn_locals, v_syn_global, t_normal, t_abnormal):
        """
        Full VTAM forward pass.

        Returns:
            p_map:   [B, 1, H, W] — pixel-level anomaly map
            s_final: [B]          — image-level anomaly score
        """
        p_map = self.compute_anomaly_map(
            v_syn_locals, v_syn_global, t_normal, t_abnormal
        )
        s_final = self.compute_score(
            v_syn_global, t_normal, t_abnormal, p_map
        )
        p_map = p_map.clamp(1e-6, 1 - 1e-6)
        anomaly_logits = torch.logit(p_map)
        return anomaly_logits, s_final
