"""
Loss Functions for SSVP — §3.4, Eq. 20-22

Total loss:
    L_total = L_Seg + L_Class + λ₁ · L_VAE + λ₂ · L_reg

Components:
    L_Seg   : Focal Loss (pixel-level segmentation, handles class imbalance)
    L_Class : BCE Loss (image-level anomaly classification)
    L_VAE   : ELBO (reconstruction + KL divergence for VAE regularization)
    L_reg   : Margin Regularization (prevents semantic drift of conditioned prompts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss — Eq. 20

    Addresses severe imbalance between normal and anomalous pixels.

    L_Seg = -1/(H*W) Σ_{h,w} [Y · (1-p)^γ · log(p) + (1-Y) · p^γ · log(1-p)]

    Args:
        gamma: Focusing parameter (2.0) — down-weights easy examples
        alpha: Optional class balance weight
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        """
        Args:
            logits: [B, 1, H, W] — predicted anomaly logits
            target: [B, 1, H, W] — ground truth mask ∈ {0, 1}
        Returns:
            loss: scalar
        """
        target = target.float()
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        # Focal modulation in probability space
        p_t = target * probs + (1 - target) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.alpha is not None:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = alpha_t * loss

        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss over probabilities."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prob, target):
        target = target.float()
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class ProOrientedLoss(nn.Module):
    """
    Differentiable PRO-oriented surrogate for anomaly localization.

    Encourages high overlap on anomalous regions while constraining activation
    on normal pixels using a target false-positive rate.

    L_pro = (1 - mean(p | y=1)) + λ_fpr * max(0, mean(p | y=0) - fpr_target)

    Args:
        fpr_target: Desired upper bound on average score in normal regions.
        fpr_penalty: Weight for violating the FPR target.
        eps: Numerical stability term.
    """

    def __init__(self, fpr_target=0.05, fpr_penalty=4.0, eps=1e-6):
        super().__init__()
        self.fpr_target = float(fpr_target)
        self.fpr_penalty = float(fpr_penalty)
        self.eps = float(eps)

    def forward(self, prob, target):
        target = target.float()
        normal = 1.0 - target

        # Per-sample anomalous-region overlap surrogate.
        pos_mass = (prob * target).sum(dim=(1, 2, 3))
        pos_norm = target.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        region_overlap = pos_mass / pos_norm

        # Per-sample normal-region activation surrogate (soft FPR proxy).
        neg_mass = (prob * normal).sum(dim=(1, 2, 3))
        neg_norm = normal.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        region_fpr = neg_mass / neg_norm

        overlap_term = 1.0 - region_overlap
        fpr_term = F.relu(region_fpr - self.fpr_target)

        return (overlap_term + self.fpr_penalty * fpr_term).mean()


class VAELoss(nn.Module):
    """
    VAE Loss (ELBO) — Eq. 9

    L_VAE = ‖V_syn^global - D_θ(z)‖² + β · D_KL(q_φ(z|V) ‖ N(0,I))

    The reconstruction term ensures the latent space captures meaningful visual
    information. The KL term regularizes toward a standard normal prior.

    Args:
        beta: KL divergence weight (0.1)
    """

    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, v_original, v_recon, mu, logvar):
        """
        Args:
            v_original: [B, D] — original V_syn^global
            v_recon:    [B, D] — reconstructed feature from VAE decoder
            mu:         [B, d_latent] — latent mean
            logvar:     [B, d_latent] — latent log-variance

        Returns:
            loss: scalar (reconstruction + weighted KL)
            recon_loss: scalar (for logging)
            kl_loss: scalar (for logging)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(v_recon, v_original.detach())

        # KL divergence: D_KL(N(μ, σ²) ‖ N(0, I))
        # = -0.5 * Σ(1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )

        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss


class MarginRegularization(nn.Module):
    """
    Margin-Based Semantic Regularization — Eq. 12

    L_reg = max(0, ξ - cos_sim(T_final, sg(T_init)))

    Prevents the vision-conditioned prompts from drifting too far from their
    initial semantics. Only penalizes when cosine similarity drops below ξ.
    This defines a "semantic neighborhood" within which the model can freely
    optimize for anomaly detection.

    Args:
        xi: Cosine similarity threshold (0.85)
    """

    def __init__(self, xi=0.85):
        super().__init__()
        self.xi = xi

    def forward(self, t_final, t_init):
        """
        Args:
            t_final: [B, n_prompts, seq_len, D] or [*, D] — conditioned prompts
            t_init:  [n_prompts, seq_len, D] or [*, D] — initial prompts (stop-gradient)

        Returns:
            loss: scalar
        """
        # Flatten to [*, D] for cosine similarity computation
        t_f = t_final.reshape(-1, t_final.shape[-1])
        t_i = t_init.reshape(-1, t_init.shape[-1])

        # Expand t_init if batch dimension is missing
        if t_f.shape[0] != t_i.shape[0]:
            # t_init is [n*seq, D], t_final is [B*n*seq, D]
            B = t_f.shape[0] // t_i.shape[0]
            t_i = t_i.unsqueeze(0).expand(B, -1, -1).reshape(-1, t_i.shape[-1])

        # Cosine similarity with stop-gradient on T_init
        cos_sim = F.cosine_similarity(t_f, t_i.detach(), dim=-1)  # [*]

        # Margin loss: penalize only when similarity < ξ
        loss = F.relu(self.xi - cos_sim).mean()

        return loss


class SSVPLoss(nn.Module):
    """
    Combined SSVP Training Objective — Eq. 22

    L_total = L_Seg + L_Class + λ₁ · L_VAE + λ₂ · L_reg

    Args:
        config: loss section of config dict
    """

    def __init__(self, config):
        super().__init__()
        cfg = config["loss"]

        self.focal_loss = FocalLoss(gamma=cfg["focal_gamma"])
        self.dice_loss = DiceLoss()
        self.pro_loss = ProOrientedLoss(
            fpr_target=float(cfg.get("pro_fpr_target", 0.05)),
            fpr_penalty=float(cfg.get("pro_fpr_penalty", 4.0)),
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.vae_loss = VAELoss(beta=cfg["beta_kl"])
        self.margin_reg = MarginRegularization(xi=cfg["xi_margin"])

        self.lambda_vae = cfg["lambda_vae"]
        self.lambda_reg = cfg["lambda_reg"]
        self.lambda_pro = float(cfg.get("lambda_pro", 0.0))
        self.lambda_denoise = float(cfg.get("lambda_denoise", 0.0))
        self.dice_weight = float(cfg.get("dice_weight", 0.3))

    def forward(self, outputs, targets):
        """
        Compute total SSVP loss.

        Args:
            outputs: dict from SSVP.forward() containing:
                - anomaly_map:  [B, 1, H, W]
                - anomaly_score: [B]
                - mu, logvar, v_syn_global, v_recon
                - t_final_normal, t_final_abnormal
                - t_init_normal, t_init_abnormal

            targets: dict containing:
                - mask:  [B, 1, H, W] — ground truth segmentation mask
                - label: [B] — image-level binary label (0=normal, 1=anomalous)

        Returns:
            total_loss: scalar
            loss_dict:  dict of individual loss components (for logging)
        """
        # ── L_Seg: Focal loss for pixel-level segmentation (Eq. 20) ──
        mask_target = targets["mask"]
        if outputs["anomaly_map"].shape[-2:] != mask_target.shape[-2:]:
            mask_target = F.interpolate(
                mask_target.float(), size=outputs["anomaly_map"].shape[-2:],
                mode="nearest"
            )
        logits = outputs["anomaly_map"]
        l_focal = self.focal_loss(logits, mask_target)
        l_dice = self.dice_loss(torch.sigmoid(logits), mask_target)
        l_pro = self.pro_loss(torch.sigmoid(logits), mask_target)
        l_seg = l_focal + self.dice_weight * l_dice + self.lambda_pro * l_pro

        # ── L_Class: BCE for image-level classification (Eq. 21) ──
        l_class = self.bce_loss(
            outputs["anomaly_score"],
            targets["label"].float()
        )

        # ── L_VAE: ELBO for VAE regularization (Eq. 9) ──
        l_vae, l_recon, l_kl = self.vae_loss(
            outputs["v_syn_global"],
            outputs["v_recon"],
            outputs["mu"],
            outputs["logvar"],
        )

        # ── L_reg: Margin regularization for semantic consistency (Eq. 12) ──
        l_reg_normal = self.margin_reg(
            outputs["t_final_normal"], outputs["t_init_normal"]
        )
        l_reg_abnormal = self.margin_reg(
            outputs["t_final_abnormal"], outputs["t_init_abnormal"]
        )
        l_reg = (l_reg_normal + l_reg_abnormal) / 2.0

        # ── Optional denoiser reconstruction supervision ──
        l_denoise = outputs.get("denoise_recon_loss", None)
        if l_denoise is None:
            l_denoise = torch.zeros_like(l_seg)

        # ── Total Loss (Eq. 22 + optional denoiser term) ──
        total_loss = (
            l_seg
            + l_class
            + self.lambda_vae * l_vae
            + self.lambda_reg * l_reg
            + self.lambda_denoise * l_denoise
        )

        loss_dict = {
            "total": total_loss.item(),
            "seg": l_seg.item(),
            "seg_focal": l_focal.item(),
            "seg_dice": l_dice.item(),
            "seg_pro": l_pro.item(),
            "class": l_class.item(),
            "vae": l_vae.item(),
            "vae_recon": l_recon.item(),
            "vae_kl": l_kl.item(),
            "reg": l_reg.item(),
            "denoise": l_denoise.item(),
        }

        return total_loss, loss_dict
