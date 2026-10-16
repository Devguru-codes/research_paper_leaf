import torch

from src.data.gpu_aug import gpu_augment, procedural_background
from src.models.esformer import ESFormer


def test_efficientnet_backbone_all_variants_forward():
    for variant, token in [("plain", None), ("global", "ap"), ("strip", None), ("esa", "none"), ("slabstat", "ap")]:
        m = ESFormer(30, "efficientnet_b0", variant, pretrained=False, img_size=128, grid=32, K=8, T=4,
                     token=token).eval()
        x = torch.randn(2, 3, 128, 128)
        mask = torch.zeros(2, 128, 128)
        mask[:, 32:96, 40:90] = 1
        logits, vein = m(x, mask)
        assert logits.shape == (2, 30) and torch.isfinite(logits).all()


def test_patchpool_ignores_the_mask():
    m = ESFormer(30, "convnext_tiny", "patchpool", pretrained=False, img_size=128, grid=32).eval()
    x = torch.randn(1, 3, 128, 128)
    m1, m2 = torch.zeros(1, 128, 128), torch.zeros(1, 128, 128)
    m1[:, 32:96, 32:96] = 1
    m2[:, 10:40, 60:120] = 1
    with torch.no_grad():
        f = m.backbone(x)
        a = m.fusion(f["patch"], f["hw"], None, None, None, torch.ones(1, f["patch"].shape[1]))
        assert torch.allclose(m(x, m1)[0], m(x, m2)[0], atol=1e-5)
    assert torch.isfinite(a).all()


def test_dinov2_backbone_forward():
    m = ESFormer(30, "dinov2_small", "maskpool", pretrained=False, img_size=384, grid=32).eval()
    x = torch.randn(1, 3, 384, 384)
    mask = torch.zeros(1, 384, 384)
    mask[:, 100:300, 120:260] = 1
    with torch.no_grad():
        logits, _ = m(x, mask)
    assert logits.shape == (1, 30) and torch.isfinite(logits).all()


def test_maskpool_uses_only_leaf_patches():
    """The mask-pooling control must ignore everything outside the leaf mask."""
    m = ESFormer(30, "convnext_tiny", "maskpool", pretrained=False, img_size=128, grid=32).eval()
    mask = torch.zeros(1, 128, 128)
    mask[:, 32:96, 32:96] = 1
    x = torch.randn(1, 3, 128, 128)
    f = m.backbone(x)
    pm = torch.nn.functional.interpolate(mask[:, None], size=f["hw"], mode="area").flatten(1)
    a = m.fusion(f["patch"], f["hw"], None, None, None, pm)
    patch = f["patch"].clone()
    patch[:, pm[0] == 0] += 5.0  # change only background patches
    b = m.fusion(patch, f["hw"], None, None, None, pm)
    assert torch.allclose(a, b, atol=1e-5)
    logits, _ = m(x, mask)
    assert logits.shape == (1, 30) and torch.isfinite(logits).all()


def test_strong_augmentation_shapes_and_background():
    g = torch.Generator().manual_seed(0)
    bg = procedural_background(4, 64, "cpu", g)
    assert bg.shape == (4, 3, 64, 64) and bg.min() >= 0 and bg.max() <= 1
    x = torch.full((4, 3, 96, 96), 255, dtype=torch.uint8)
    m = torch.zeros(4, 96, 96, dtype=torch.uint8)
    m[:, 30:60, 30:60] = 255
    v = torch.zeros(4, 96, 96, dtype=torch.uint8)
    xa, ma, va = gpu_augment(x, m, v, 64, generator=torch.Generator().manual_seed(1), strong=True)
    assert xa.shape == (4, 3, 64, 64) and ma.shape == (4, 64, 64) and torch.isfinite(xa).all()
    xp, _, _ = gpu_augment(x, m, v, 64, generator=torch.Generator().manual_seed(1), strong=False)
    # with strong augmentation the (white) background is replaced for most samples
    assert (xa - xp).abs().mean() > 0.05
