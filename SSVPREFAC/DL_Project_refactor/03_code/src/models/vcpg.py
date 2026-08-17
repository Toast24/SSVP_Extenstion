"""
Vision-Conditioned Prompt Generator (VCPG) — §3.2

Generates dynamic, vision-conditioned text prompts that adapt to unseen anomaly
patterns by injecting visual latent biases into learnable text embeddings.

Three sub-components:
    A) Structured Learnable Prompt Embeddings (§3.2.1)
       - Shared [V]_bg background vectors + state-specific [V]_normal/[V]_abnormal
       
    B) Variational Visual Modeling (§3.2.2)
       - VAE encodes V_syn^global → latent z via reparameterization
       - Text-Latent Cross-Attention: T_init queries z to produce δ_inj
       
    C) Gated Injection with Margin Regularization (§3.2.3)
       - T_final = LayerNorm(T_init + α · δ_inj)
       - L_reg = max(0, ξ - cos_sim(T_final, sg(T_init)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PromptBank(nn.Module):
    """
    Structured Learnable Prompt Embeddings — §3.2.1

    Constructs decoupled prompt templates:
        t = [V]_bg, [V]_state, [CLASS]

    The [V]_bg vectors are SHARED across normal/abnormal prompts to learn
    domain-invariant environmental semantics. [V]_state vectors are separate
    for each state to learn discriminative features.

    Args:
        n_normal:       Number of normal prompts (3)
        n_abnormal:     Number of abnormal prompts (3)
        bg_ctx_len:     Background context length (4 tokens)
        state_ctx_len:  State context length (4 tokens)
        d_text:         Text embedding dimension (768)
    """

    def __init__(self, n_normal=3, n_abnormal=3, bg_ctx_len=4,
                 state_ctx_len=4, d_text=768):
        super().__init__()
        self.n_normal = n_normal
        self.n_abnormal = n_abnormal
        self.bg_ctx_len = bg_ctx_len
        self.state_ctx_len = state_ctx_len
        self.d_text = d_text

        # Shared background vectors: domain-invariant context
        # Shape: [bg_ctx_len, d_text]
        self.bg_vectors = nn.Parameter(
            torch.randn(bg_ctx_len, d_text) * 0.02
        )

        # Normal state vectors: one set per normal prompt
        # Shape: [n_normal, state_ctx_len, d_text]
        self.normal_state = nn.Parameter(
            torch.randn(n_normal, state_ctx_len, d_text) * 0.02
        )

        # Abnormal state vectors: one set per abnormal prompt
        # Shape: [n_abnormal, state_ctx_len, d_text]
        self.abnormal_state = nn.Parameter(
            torch.randn(n_abnormal, state_ctx_len, d_text) * 0.02
        )

    def get_prompt_embeddings(self, class_token_embedding=None):
        """
        Build full prompt embedding sequences for all prompts.

        Args:
            class_token_embedding: [D_text] — embedding for [CLASS] token
                                   If None, a zero vector is used.

        Returns:
            normal_prompts:   [n_normal, seq_len, d_text]
            abnormal_prompts: [n_abnormal, seq_len, d_text]

            where seq_len = bg_ctx_len + state_ctx_len (+ 1 if class token)
        """
        prompts_normal = []
        prompts_abnormal = []

        # Build normal prompts: [bg] + [state_normal_i] + [class]
        for i in range(self.n_normal):
            parts = [self.bg_vectors, self.normal_state[i]]
            if class_token_embedding is not None:
                parts.append(class_token_embedding.unsqueeze(0))
            prompt = torch.cat(parts, dim=0)  # [seq_len, d_text]
            prompts_normal.append(prompt)

        # Build abnormal prompts: [bg] + [state_abnormal_i] + [class]
        for i in range(self.n_abnormal):
            parts = [self.bg_vectors, self.abnormal_state[i]]
            if class_token_embedding is not None:
                parts.append(class_token_embedding.unsqueeze(0))
            prompt = torch.cat(parts, dim=0)  # [seq_len, d_text]
            prompts_abnormal.append(prompt)

        normal_prompts = torch.stack(prompts_normal, dim=0)     # [n_normal, seq, d]
        abnormal_prompts = torch.stack(prompts_abnormal, dim=0)  # [n_abnormal, seq, d]

        return normal_prompts, abnormal_prompts


class VAEModule(nn.Module):
    """
    Variational Autoencoder for visual latent modeling — §3.2.2, Eq. 7-9

    Encodes V_syn^global into a latent Gaussian distribution, samples z via
    reparameterization, and decodes for reconstruction loss.

    The latent variable z serves as the "Visual Latent Bias" encoding the
    global anomaly distribution of the input image.

    Args:
        d_visual:  Input visual feature dim (768, from HSVS)
        d_latent:  Latent space dim (256)
        d_hidden:  Hidden layer dim for encoder/decoder MLPs
    """

    def __init__(self, d_visual=768, d_latent=256, d_hidden=512):
        super().__init__()
        self.d_latent = d_latent

        # Encoder: V_syn^global → (μ, log σ²)  (Eq. 7)
        self.encoder_shared = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
        )
        self.fc_mu = nn.Linear(d_hidden, d_latent)       # F_μ
        self.fc_logvar = nn.Linear(d_hidden, d_latent)    # F_σ

        # Decoder: z → V̂  (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_visual),
        )

    def encode(self, x):
        """
        Encode visual feature to latent distribution parameters.

        Args:
            x: [B, d_visual] — V_syn^global
        Returns:
            mu:     [B, d_latent]
            logvar: [B, d_latent]  (log σ²)
        """
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick (Eq. 8):
            z = μ + ε ⊙ exp(log σ² / 2),  ε ~ N(0, I)

        Args:
            mu:     [B, d_latent]
            logvar: [B, d_latent]
        Returns:
            z: [B, d_latent]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # During inference, use the mean directly for deterministic output
            z = mu
        return z

    def decode(self, z):
        """
        Decode latent variable to reconstruct visual feature.

        Args:
            z: [B, d_latent]
        Returns:
            x_recon: [B, d_visual]
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full VAE forward: encode → reparameterize → decode.

        Args:
            x: [B, d_visual]
        Returns:
            x_recon: [B, d_visual]
            mu:      [B, d_latent]
            logvar:  [B, d_latent]
            z:       [B, d_latent]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


class TextLatentCrossAttention(nn.Module):
    """
    Text-Latent Cross-Attention — §3.2.2, Eq. 10

    Allows text embeddings (T_init) to dynamically "retrieve" and integrate
    visual biases from the latent variable z.

    Q = T_init · W_Q   (text queries)
    K = z · W_K         (visual latent keys)
    V = z · W_V         (visual latent values)
    δ_inj = Softmax(Q · K^T / √d_k) · V

    Args:
        d_text:   Text embedding dim (768)
        d_latent: Latent variable dim (256)
        d_k:      Key/Query projection dim (256)
        num_heads: Number of attention heads (4)
    """

    def __init__(self, d_text=768, d_latent=256, d_k=256, num_heads=4):
        super().__init__()
        self.d_k = d_k
        self.num_heads = num_heads
        self.head_dim = d_k // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(d_text, d_k, bias=False)
        self.W_K = nn.Linear(d_latent, d_k, bias=False)
        self.W_V = nn.Linear(d_latent, d_text, bias=False)  # Projects back to text dim

        self.out_proj = nn.Linear(d_text, d_text)
        self.ln = nn.LayerNorm(d_text)

    def forward(self, t_init, z_tokens):
        """
        Cross-attention between text embeddings and visual latent.

        Args:
            t_init: [B, N_prompts, d_text] — initial text embeddings
            z_tokens: [B, M, d_latent] — visual latent token sequence

        Returns:
            delta_inj: [B, N_prompts, d_text] — visual bias residual
        """
        B, N, D = t_init.shape
        M = z_tokens.shape[1]

        # Project to attention space
        Q = self.W_Q(t_init)      # [B, N, d_k]
        K = self.W_K(z_tokens)    # [B, M, d_k]
        V = self.W_V(z_tokens)    # [B, M, d_text]

        # Multi-head reshape
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, hd]
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, hd]

        # Split text-dim values per head for proper attention composition
        Vh = V.view(B, M, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, H, M, D/H]

        # Attention over latent tokens
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N, M]
        attn = F.softmax(attn, dim=-1)

        # Weighted value aggregation
        context = torch.matmul(attn, Vh)  # [B, H, N, D/H]
        context = context.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        delta_inj = self.out_proj(context)
        delta_inj = self.ln(delta_inj)

        return delta_inj


class VCPG(nn.Module):
    """
    Vision-Conditioned Prompt Generator — §3.2

    Full pipeline:
        1. Get initial prompt embeddings from PromptBank
        2. Encode V_syn^global via VAE → latent z
        3. Cross-attention: T_init queries z → δ_inj
        4. Gated injection: T_final = LN(T_init + α · δ_inj)

    Args:
        config: vcpg section of config dict
    """

    def __init__(self, config):
        super().__init__()
        cfg = config["vcpg"]

        self.d_text = cfg["d_text"]
        self.d_latent = cfg["d_latent"]
        d_proj = config["hsvs"]["d_proj"]

        # Sub-component A: Learnable Prompt Bank
        self.prompt_bank = PromptBank(
            n_normal=cfg["n_normal_prompts"],
            n_abnormal=cfg["n_abnormal_prompts"],
            bg_ctx_len=cfg["bg_context_len"],
            state_ctx_len=cfg["state_context_len"],
            d_text=cfg["d_text"],
        )

        # Sub-component B: VAE for visual latent modeling
        self.vae = VAEModule(
            d_visual=d_proj,       # Input is V_syn^global from HSVS (768)
            d_latent=cfg["d_latent"],
            d_hidden=512,
        )

        # Sub-component B (cont): Text-Latent Cross-Attention
        self.cross_attn = TextLatentCrossAttention(
            d_text=cfg["d_text"],
            d_latent=cfg["d_latent"],
            d_k=cfg["d_latent"],
            num_heads=4,
        )

        # Expand a single latent vector into multiple latent tokens so
        # cross-attention can learn meaningful token selection.
        self.num_latent_tokens = int(cfg.get("num_latent_tokens", 1))
        self.latent_tokenizer = nn.Sequential(
            nn.Linear(cfg["d_latent"], cfg["d_latent"] * self.num_latent_tokens),
            nn.GELU(),
        )

        # Sub-component C: Gated injection
        self.alpha = nn.Parameter(torch.tensor(cfg["alpha_init"]))  # Init 0.0
        self.ln_final = nn.LayerNorm(cfg["d_text"])

        # Projection: if d_proj != d_text, need alignment for prompt space
        if d_proj != cfg["d_text"]:
            self.text_proj = nn.Linear(cfg["d_text"], d_proj, bias=False)
        else:
            self.text_proj = nn.Identity()

        # Prompt pooling strategy defaults to run21-compatible mean pooling.
        self.prompt_pooling = str(cfg.get("prompt_pooling", "mean")).strip().lower()
        if self.prompt_pooling not in {"mean", "weighted"}:
            raise ValueError(
                f"Unsupported vcpg.prompt_pooling '{self.prompt_pooling}'. "
                "Supported values: mean, weighted"
            )

        self.prompt_pool_score = None
        if self.prompt_pooling == "weighted":
            hidden_pool = max(cfg["d_text"] // 2, 64)
            self.prompt_pool_score = nn.Sequential(
                nn.Linear(cfg["d_text"], hidden_pool),
                nn.GELU(),
                nn.Linear(hidden_pool, 1),
            )

    def forward(self, v_syn_global, class_token_embedding=None):
        """
        Generate vision-conditioned prompt embeddings.

        Args:
            v_syn_global:          [B, d_proj] — global synergistic feature from HSVS
            class_token_embedding: [d_text] — optional class name embedding

        Returns:
            t_final_normal:    [B, n_normal, seq_len, d_text]  — conditioned normal prompts
            t_final_abnormal:  [B, n_abnormal, seq_len, d_text] — conditioned abnormal prompts
            t_init_normal:     [n_normal, seq_len, d_text] — initial normal prompts
            t_init_abnormal:   [n_abnormal, seq_len, d_text] — initial abnormal prompts
            vae_outputs:       dict with mu, logvar, z, v_recon
        """
        B = v_syn_global.shape[0]

        # ── Step 1: Get initial prompt embeddings ──
        t_init_normal, t_init_abnormal = self.prompt_bank.get_prompt_embeddings(
            class_token_embedding
        )
        # t_init_*: [n_*, seq_len, d_text]

        # ── Step 2: VAE encoding of visual features ──
        v_recon, mu, logvar, z = self.vae(v_syn_global)
        # z: [B, d_latent]
        z_tokens = self.latent_tokenizer(z).view(B, self.num_latent_tokens, self.d_latent)

        # ── Step 3: Cross-attention — text queries visual latent ──
        # Process normal prompts
        # Expand t_init for batch: [n_normal, seq, d] → [B, n_normal*seq, d]
        n_norm = t_init_normal.shape[0]
        seq_len = t_init_normal.shape[1]
        t_init_n_batch = t_init_normal.unsqueeze(0).expand(B, -1, -1, -1)
        t_init_n_flat = t_init_n_batch.reshape(B, n_norm * seq_len, self.d_text)

        delta_inj_n = self.cross_attn(t_init_n_flat, z_tokens)  # [B, n_norm*seq, d_text]
        delta_inj_n = delta_inj_n.view(B, n_norm, seq_len, self.d_text)

        # Process abnormal prompts
        n_abn = t_init_abnormal.shape[0]
        t_init_a_batch = t_init_abnormal.unsqueeze(0).expand(B, -1, -1, -1)
        t_init_a_flat = t_init_a_batch.reshape(B, n_abn * seq_len, self.d_text)

        delta_inj_a = self.cross_attn(t_init_a_flat, z_tokens)  # [B, n_abn*seq, d_text]
        delta_inj_a = delta_inj_a.view(B, n_abn, seq_len, self.d_text)

        # ── Step 4: Gated injection (Eq. 11) ──
        t_final_normal = self.ln_final(
            t_init_normal.unsqueeze(0).expand(B, -1, -1, -1) + self.alpha * delta_inj_n
        )  # [B, n_normal, seq_len, d_text]

        t_final_abnormal = self.ln_final(
            t_init_abnormal.unsqueeze(0).expand(B, -1, -1, -1) + self.alpha * delta_inj_a
        )  # [B, n_abnormal, seq_len, d_text]

        vae_outputs = {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "v_recon": v_recon,
        }

        return (t_final_normal, t_final_abnormal,
                t_init_normal, t_init_abnormal,
                vae_outputs)

    def get_aggregated_prompt_features(self, t_final_normal, t_final_abnormal):
        """
        Aggregate multi-prompt embeddings into single normal/abnormal representations.
        Uses mean across prompts and sequence positions.

        Args:
            t_final_normal:   [B, n_normal, seq_len, d_text]
            t_final_abnormal: [B, n_abnormal, seq_len, d_text]

        Returns:
            normal_feat:   [B, d_text] — aggregated normal embedding
            abnormal_feat: [B, d_text] — aggregated abnormal embedding
        """
        if self.prompt_pooling == "mean":
            normal_feat = t_final_normal.mean(dim=(1, 2))
            abnormal_feat = t_final_abnormal.mean(dim=(1, 2))
            return normal_feat, abnormal_feat

        def _weighted_pool(t_prompt):
            bsz, n_prompts, seq_len, d_dim = t_prompt.shape
            tokens = t_prompt.reshape(bsz, n_prompts * seq_len, d_dim)
            scores = self.prompt_pool_score(tokens).squeeze(-1)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            return (tokens * weights).sum(dim=1)

        normal_feat = _weighted_pool(t_final_normal)      # [B, d_text]
        abnormal_feat = _weighted_pool(t_final_abnormal)  # [B, d_text]
        return normal_feat, abnormal_feat
