"""
SSVP Shape Verification Test

Validates that all modules produce correct tensor shapes matching the paper's
specifications. Runs on CPU with small dummy inputs (no GPU required).
"""

import sys
import torch
import torch.nn as nn

from path_utils import default_config_path, ensure_import_paths

ensure_import_paths()

from utils import load_config


DEFAULT_CONFIG_PATH = default_config_path()


def test_atf_block():
    """Test Adaptive Token Features Fusion block shapes."""
    from models.hsvs import ATFBlock

    B, N = 2, 1369  # batch=2, 37×37 patches
    D_clip, D_dino, D_proj = 1024, 1024, 768

    atf = ATFBlock(d_clip=D_clip, d_dino=D_dino, d_head=128,
                   num_heads=6, d_proj=D_proj, mlp_ratio=4)

    feat_clip = torch.randn(B, N, D_clip)
    feat_dino = torch.randn(B, N, D_dino)

    v_syn = atf(feat_clip, feat_dino)

    assert v_syn.shape == (B, N, D_proj), \
        f"ATF output shape mismatch: {v_syn.shape} != ({B}, {N}, {D_proj})"
    print(f"  ✓ ATFBlock: [{B}, {N}, {D_clip}] + [{B}, {N}, {D_dino}] → [{B}, {N}, {D_proj}]")
    return True


def test_hsvs():
    """Test Hierarchical Semantic-Visual Synergy shapes."""
    from models.hsvs import HSVS

    B, N = 2, 1369
    D_clip, D_dino, D_proj = 1024, 1024, 768

    hsvs = HSVS(d_clip=D_clip, d_dino=D_dino, d_proj=D_proj,
                d_head=128, num_heads=6, num_layers=4, mlp_ratio=4)

    clip_global = torch.randn(B, D_clip)
    clip_locals = [torch.randn(B, N, D_clip) for _ in range(4)]
    dino_global = torch.randn(B, D_dino)
    dino_locals = [torch.randn(B, N, D_dino) for _ in range(4)]

    v_syn_global, v_syn_locals = hsvs(clip_global, clip_locals, dino_global, dino_locals)

    assert v_syn_global.shape == (B, D_proj), \
        f"Global shape mismatch: {v_syn_global.shape}"
    assert len(v_syn_locals) == 4
    for l, v in enumerate(v_syn_locals):
        assert v.shape == (B, N, D_proj), \
            f"Local layer {l} shape mismatch: {v.shape}"

    print(f"  ✓ HSVS: global=[{B}, {D_proj}], local=4×[{B}, {N}, {D_proj}]")
    return True


def test_vcpg():
    """Test Vision-Conditioned Prompt Generator shapes."""
    from models.vcpg import VCPG

    config = load_config(DEFAULT_CONFIG_PATH)
    B, D_proj = 2, 768

    vcpg = VCPG(config)
    v_syn_global = torch.randn(B, D_proj)

    (t_final_normal, t_final_abnormal,
     t_init_normal, t_init_abnormal,
     vae_outputs) = vcpg(v_syn_global)

    n_norm = config["vcpg"]["n_normal_prompts"]
    n_abn = config["vcpg"]["n_abnormal_prompts"]
    seq_len = config["vcpg"]["bg_context_len"] + config["vcpg"]["state_context_len"]
    d_text = config["vcpg"]["d_text"]

    assert t_final_normal.shape == (B, n_norm, seq_len, d_text), \
        f"Normal prompt shape: {t_final_normal.shape}"
    assert t_final_abnormal.shape == (B, n_abn, seq_len, d_text), \
        f"Abnormal prompt shape: {t_final_abnormal.shape}"
    assert vae_outputs["z"].shape == (B, config["vcpg"]["d_latent"]), \
        f"Latent z shape: {vae_outputs['z'].shape}"
    assert vae_outputs["mu"].shape == (B, config["vcpg"]["d_latent"])
    assert vae_outputs["v_recon"].shape == (B, D_proj)

    # Test aggregation
    norm_agg, abn_agg = vcpg.get_aggregated_prompt_features(
        t_final_normal, t_final_abnormal
    )
    assert norm_agg.shape == (B, d_text), f"Agg normal shape: {norm_agg.shape}"
    assert abn_agg.shape == (B, d_text), f"Agg abnormal shape: {abn_agg.shape}"

    print(f"  ✓ VCPG: T_final_norm=[{B}, {n_norm}, {seq_len}, {d_text}], "
          f"T_final_abn=[{B}, {n_abn}, {seq_len}, {d_text}], z=[{B}, {config['vcpg']['d_latent']}]")
    return True


def test_vtam():
    """Test Visual-Text Anomaly Mapper shapes."""
    from models.vtam import VTAM

    B, N, D = 2, 1369, 768
    H = W = 37

    vtam = VTAM(d_proj=D, num_layers=4, tau=0.07, gamma=0.5)

    v_syn_locals = [torch.randn(B, N, D) for _ in range(4)]
    v_syn_global = torch.randn(B, D)
    t_normal = torch.randn(B, D)
    t_abnormal = torch.randn(B, D)

    p_map, s_final = vtam(v_syn_locals, v_syn_global, t_normal, t_abnormal)

    assert p_map.shape == (B, 1, H, W), f"Anomaly map shape: {p_map.shape}"
    assert s_final.shape == (B,), f"Score shape: {s_final.shape}"

    print(f"  ✓ VTAM: P_map=[{B}, 1, {H}, {W}], S_final=[{B}]")
    return True


def test_losses():
    """Test loss function shapes and computation."""
    from models.losses import SSVPLoss

    config = load_config(DEFAULT_CONFIG_PATH)
    criterion = SSVPLoss(config)

    B, H, W, D = 2, 37, 37, 768
    d_text = config["vcpg"]["d_text"]
    n_norm = config["vcpg"]["n_normal_prompts"]
    n_abn = config["vcpg"]["n_abnormal_prompts"]
    seq_len = config["vcpg"]["bg_context_len"] + config["vcpg"]["state_context_len"]

    outputs = {
        "anomaly_map": torch.rand(B, 1, H, W, requires_grad=True),
        "anomaly_score": torch.randn(B, requires_grad=True),
        "mu": torch.randn(B, 256),
        "logvar": torch.randn(B, 256),
        "v_syn_global": torch.randn(B, D),
        "v_recon": torch.randn(B, D),
        "t_final_normal": torch.randn(B, n_norm, seq_len, d_text),
        "t_final_abnormal": torch.randn(B, n_abn, seq_len, d_text),
        "t_init_normal": torch.randn(n_norm, seq_len, d_text),
        "t_init_abnormal": torch.randn(n_abn, seq_len, d_text),
    }

    targets = {
        "mask": torch.randint(0, 2, (B, 1, H, W)).float(),
        "label": torch.randint(0, 2, (B,)).float(),
    }

    total_loss, loss_dict = criterion(outputs, targets)

    assert total_loss.dim() == 0, "Total loss should be scalar"
    assert total_loss.requires_grad, "Total loss should require grad"
    assert set(["total", "seg", "class", "vae", "reg"]).issubset(loss_dict.keys())

    print(f"  ✓ SSVPLoss: total={total_loss.item():.4f} "
          f"(seg={loss_dict['seg']:.4f}, cls={loss_dict['class']:.4f}, "
          f"vae={loss_dict['vae']:.4f}, reg={loss_dict['reg']:.4f})")
    return True


def test_full_trainable_pipeline():
    """
    Test the trainable modules together (without frozen backbones).
    Simulates the forward pass with dummy backbone outputs.
    """
    from models.hsvs import HSVS
    from models.vcpg import VCPG
    from models.vtam import VTAM
    from models.losses import SSVPLoss

    config = load_config(DEFAULT_CONFIG_PATH)

    B, N = 2, 1369
    D_clip, D_dino, D_proj = 1024, 1024, 768

    # Initialize modules
    hsvs = HSVS(D_clip, D_dino, D_proj, d_head=128, num_heads=6, num_layers=4)
    vcpg = VCPG(config)
    vtam = VTAM(d_proj=D_proj, num_layers=4)
    criterion = SSVPLoss(config)

    # Simulate backbone outputs
    clip_global = torch.randn(B, D_clip)
    clip_locals = [torch.randn(B, N, D_clip) for _ in range(4)]
    dino_global = torch.randn(B, D_dino)
    dino_locals = [torch.randn(B, N, D_dino) for _ in range(4)]

    # Forward pass
    v_syn_global, v_syn_locals = hsvs(clip_global, clip_locals, dino_global, dino_locals)

    (t_final_n, t_final_a, t_init_n, t_init_a, vae_out) = vcpg(v_syn_global)
    t_norm_agg, t_abn_agg = vcpg.get_aggregated_prompt_features(t_final_n, t_final_a)

    p_map, s_final = vtam(v_syn_locals, v_syn_global, t_norm_agg, t_abn_agg)

    # Loss computation
    outputs = {
        "anomaly_map": p_map,
        "anomaly_score": s_final,
        "mu": vae_out["mu"],
        "logvar": vae_out["logvar"],
        "v_syn_global": v_syn_global,
        "v_recon": vae_out["v_recon"],
        "t_final_normal": t_final_n,
        "t_final_abnormal": t_final_a,
        "t_init_normal": t_init_n,
        "t_init_abnormal": t_init_a,
    }

    targets = {
        "mask": torch.randint(0, 2, (B, 1, 37, 37)).float(),
        "label": torch.randint(0, 2, (B,)).float(),
    }

    loss, loss_dict = criterion(outputs, targets)

    # Backward pass
    loss.backward()

    # Check gradients exist
    has_grads = sum(1 for p in hsvs.parameters() if p.grad is not None)
    total_trainable = sum(1 for p in hsvs.parameters() if p.requires_grad)

    print(f"  ✓ Full pipeline: loss={loss.item():.4f}, "
          f"gradients: {has_grads}/{total_trainable} HSVS params")

    # Count all trainable params
    all_params = (list(hsvs.parameters()) + list(vcpg.parameters()) +
                  list(vtam.parameters()))
    trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"  ✓ Total trainable parameters: {trainable:,}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  SSVP Shape Verification Tests")
    print("=" * 60)

    tests = [
        ("ATF Block", test_atf_block),
        ("HSVS Module", test_hsvs),
        ("VCPG Module", test_vcpg),
        ("VTAM Module", test_vtam),
        ("Loss Functions", test_losses),
        ("Full Trainable Pipeline", test_full_trainable_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
