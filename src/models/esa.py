"""Euler-Sweep Attention (ESA) and the comparison fusion modules.

ESA (token='euler', attn='slab'): one sweep token per (direction theta, slab k). The token's
query is built from the filtration increments Delta chi(theta,k) of the blade complex and of
the learned venation field; it cross-attends to backbone patch tokens with an additive bias
log(s_{theta,k}(p) * leaf(p) + eps), where s is the *same* soft slab membership that defines
Delta chi. Tokens are then mixed by a sweep transformer whose relative position bias is
cyclic in theta, and mean-pooled.

Controls (same module, different switches):
  token='stats'  -> non-topological slab statistics (area, perimeter, spread, vein mass)
  attn='global'  -> attention over all leaf patches (no slab restriction)
  weights='unit' | 'permuted' -> unit cell weights / spatially shuffled learned weights
GlobalTopoFusion: the same Delta chi grid, flattened through an MLP (global fusion baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cubical_ect import (SweepGeometry, euler_increments, frame_coords, normalise_frame,
                                    pixel_coords, slab_statistics, valuation_increments)

# number of per-slab structural features for each token mode
TOKEN_DIMS = {"euler": 4, "stats": 4, "ap": 8, "hadwiger": 12, "none": 0}


def slog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def _permute_within(a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Shuffle values of a (B,H,W) among the pixels where m > 0.5, independently per sample."""
    out = a.clone()
    for b in range(a.shape[0]):
        idx = torch.nonzero(m[b].flatten() > 0.5).squeeze(1)
        if idx.numel() > 1:
            flat = out[b].flatten()
            flat[idx] = a[b].flatten()[idx[torch.randperm(idx.numel(), device=a.device)]]
            out[b] = flat.view_as(out[b])
    return out


class StructuralWeights(nn.Module):
    """Appearance-conditioned per-pixel weights for the blade and venation complexes."""

    def __init__(self, in_ch: int, mode: str = "learned", hidden: int = 64):
        super().__init__()
        self.mode = mode
        if mode != "unit":
            self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 2, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, 2.0)  # start near unit weights (sigmoid(2)=0.88)

    def forward(self, wfeat, blade, vein):
        if self.mode == "unit":
            return blade, vein * blade
        G = blade.shape[-1]
        wfeat = wfeat.float()  # this block runs in fp32 (autocast disabled); backbones may emit fp16
        g = torch.sigmoid(self.net(F.interpolate(wfeat, size=(G, G), mode="bilinear", align_corners=False)))
        gb, gv = g[:, 0].float(), g[:, 1].float()
        if self.mode == "permuted":
            gb, gv = _permute_within(gb, blade), _permute_within(gv, blade)
        return blade * gb, vein * blade * gv


class StructuralSignals(nn.Module):
    """Per-(theta,k) structural token features (B,K,T,4) and the normalised frame.

    token='euler': [dchi_blade, dchi_vein, chi_blade(t_k), chi_vein(t_k)]
    token='stats': [area_blade, perimeter_blade, spread_blade, vein_mass]   (pilot-1 control)
    token='ap':       increments + cumulative of area and perimeter, blade and vein (no Euler; control)
    token='hadwiger': increments + cumulative of area, perimeter and Euler characteristic, blade and
                      vein (complete Hadwiger basis of planar motion-invariant valuations)
    """

    def __init__(self, wfeat_dim: int, K: int = 32, T: int = 16, R: float = 2.0, kappa: float = 4.0,
                 nbins: int = 256, token: str = "euler", weights: str = "learned",
                 use_blade: bool = True, use_vein: bool = True):
        super().__init__()
        assert token in TOKEN_DIMS
        self.token, self.use_blade, self.use_vein = token, use_blade, use_vein
        self.geo = SweepGeometry(K, T, R, kappa, nbins)
        self.weights = StructuralWeights(wfeat_dim, weights)

    def forward(self, wfeat, blade, vein):
        with torch.autocast(device_type=blade.device.type, enabled=False):
            blade, vein = blade.float(), vein.float()
            centre, scale = normalise_frame(blade)
            G = blade.shape[-1]
            coords = frame_coords(pixel_coords(G, G, blade.device), centre, scale)
            ab, av = self.weights(wfeat, blade, vein)
            ab, av = ab.float(), av.float()
            if not self.use_blade:
                ab = torch.zeros_like(ab)
            if not self.use_vein:
                av = torch.zeros_like(av)
            if self.token == "none":  # slab-only control: geometry but no structural content
                feats = coords.new_zeros(coords.shape[0], self.geo.K, self.geo.T, 0)
            elif self.token == "euler":
                db = euler_increments(ab, coords, self.geo)
                dv = euler_increments(av, coords, self.geo)
                feats = torch.stack([db, dv, db.cumsum(-1), dv.cumsum(-1)], -1)
            elif self.token in ("ap", "hadwiger"):
                vb = valuation_increments(ab, coords, self.geo)  # (B,K,T,3): area, perimeter, chi
                vv = valuation_increments(av, coords, self.geo)
                keep = [0, 1] if self.token == "ap" else [0, 1, 2]
                inc = torch.cat([vb[..., keep], vv[..., keep]], -1)
                feats = torch.cat([inc, inc.cumsum(2)], -1)
            else:
                sb = slab_statistics(ab, coords, self.geo)  # area, perimeter, spread
                h = torch.einsum("bhwc,kc->bkhw", coords, self.geo.dirs.to(coords)).flatten(2)
                vm = self.geo.slab_sums(av.flatten(1), h)
                feats = torch.cat([sb, vm[..., None]], -1)
        return slog(feats), centre, scale


class SweepBlock(nn.Module):
    """Pre-LN transformer block with a learned relative bias cyclic in theta and linear in k."""

    def __init__(self, d: int, heads: int, K: int, T: int, mlp: int = 4):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.table = nn.Parameter(torch.zeros(heads, K, 2 * T - 1))
        th = torch.arange(K).repeat_interleave(T)
        kk = torch.arange(T).repeat(K)
        dth = (th[None, :] - th[:, None]) % K
        dk = kk[None, :] - kk[:, None] + T - 1
        self.register_buffer("idx", dth * (2 * T - 1) + dk, persistent=False)  # (L, L)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        bias = self.table.flatten(1)[:, self.idx].to(q.dtype)  # (h, L, L)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])
        x = x + self.proj(y.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.n2(x))


class EulerSweepAttention(nn.Module):
    def __init__(self, patch_dim: int, wfeat_dim: int, d: int = 192, K: int = 32, T: int = 16,
                 R: float = 2.0, kappa: float = 4.0, nbins: int = 256, heads: int = 4, depth: int = 2,
                 token: str = "euler", attn: str = "slab", weights: str = "learned",
                 use_blade: bool = True, use_vein: bool = True):
        super().__init__()
        assert attn in ("slab", "global") and token in TOKEN_DIMS
        self.attn, self.K, self.T, self.heads = attn, K, T, heads
        self.signals = StructuralSignals(wfeat_dim, K, T, R, kappa, nbins, token, weights, use_blade, use_vein)
        self.geo = self.signals.geo
        nfeat = TOKEN_DIMS[token]
        self.topo_mlp = nn.Sequential(nn.Linear(nfeat, d), nn.GELU(), nn.Linear(d, d)) if nfeat else None
        self.slab_emb = nn.Parameter(torch.zeros(T, d))
        nn.init.trunc_normal_(self.slab_emb, std=0.02)
        self.kv_norm = nn.LayerNorm(patch_dim)
        self.q_proj, self.kv_proj = nn.Linear(d, d), nn.Linear(patch_dim, 2 * d)
        self.o_proj = nn.Linear(d, d)
        self.tok_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([SweepBlock(d, heads, K, T) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, patch, patch_hw, wfeat, blade, vein, patch_mask, return_attn: bool = False):
        B, P, _ = patch.shape
        K, T = self.K, self.T
        feats, centre, scale = self.signals(wfeat, blade, vein)
        if self.topo_mlp is not None:
            q = self.topo_mlp(feats.to(patch.dtype)) + self.slab_emb[None, None]  # (B,K,T,d)
        else:
            q = self.slab_emb[None, None].expand(B, K, T, -1).to(patch.dtype)
        q = q.flatten(1, 2)  # (B, L, d), L = K*T, theta-major

        # co-registered slab membership of patch centres (same relaxation as Delta chi)
        h, w = patch_hw
        pc = frame_coords(pixel_coords(h, w, patch.device), centre, scale).flatten(1, 2)  # (B,P,2)
        with torch.autocast(device_type=patch.device.type, enabled=False):
            if self.attn == "slab":
                hp = torch.einsum("bpc,kc->bkp", pc.float(), self.geo.dirs.to(pc.device, torch.float32))
                s = self.geo.slab_membership(hp)  # (B,K,P,T)
                s = s.permute(0, 1, 3, 2).reshape(B, K * T, P)
                support = s * patch_mask.float()[:, None, :]
            else:
                support = patch_mask.float()[:, None, :].expand(B, K * T, P)
            bias = torch.log(support + 1e-4)
            mass = support.sum(-1, keepdim=True)
            gate = mass / (mass + 0.5)

        d = q.shape[-1]
        hd = d // self.heads
        kv = self.kv_proj(self.kv_norm(patch)).view(B, P, 2, self.heads, hd).permute(2, 0, 3, 1, 4)
        qh = self.q_proj(q).view(B, K * T, self.heads, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(qh, kv[0], kv[1], attn_mask=bias[:, None].to(qh.dtype))
        y = self.o_proj(y.transpose(1, 2).reshape(B, K * T, d))
        x = self.tok_norm(q + gate.to(y.dtype) * y)
        for blk in self.blocks:
            x = blk(x)
        out = self.out_norm(x.mean(1))
        if return_attn:
            attn = torch.softmax((qh @ kv[0].transpose(-1, -2)) / hd ** 0.5 + bias[:, None].to(qh.dtype), -1)
            return out, dict(attn=attn.mean(1), feats=feats, support=support)
        return out


class GlobalTopoFusion(nn.Module):
    """Global fusion baseline: the full Delta chi grid -> MLP (no co-registration)."""

    def __init__(self, wfeat_dim: int, d: int = 192, K: int = 32, T: int = 16, R: float = 2.0,
                 kappa: float = 4.0, nbins: int = 256, weights: str = "learned", token: str = "euler"):
        super().__init__()
        self.signals = StructuralSignals(wfeat_dim, K, T, R, kappa, nbins, token, weights)
        n = K * T * TOKEN_DIMS[token]
        self.mlp = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, 2 * d), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.out_dim = d

    def forward(self, patch, patch_hw, wfeat, blade, vein, patch_mask, return_attn=False):
        feats, _, _ = self.signals(wfeat, blade, vein)
        out = self.mlp(feats.flatten(1).to(patch.dtype))
        return (out, dict(feats=feats)) if return_attn else out


class MaskPoolFusion(nn.Module):
    """Mask-pooling control: the mask-weighted mean of the same patch tokens over the whole leaf.
    Uses the leaf mask exactly as the localised modules do, but has no localisation and no geometry,
    so it tests whether their gain is explained by knowing where the leaf is."""

    def __init__(self, patch_dim: int, d: int = 192):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, d), nn.GELU(), nn.Linear(d, d))
        self.out_norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, patch, patch_hw, wfeat, blade, vein, patch_mask, return_attn=False):
        z = self.inp(patch)
        m = patch_mask[..., None].to(z.dtype)
        out = self.out_norm((z * m).sum(1) / m.sum(1).clamp_min(1e-3))
        return (out, {}) if return_attn else out


class StripPoolFusion(nn.Module):
    """Strip-pooling / axial baseline (cf. Hou et al., CVPR 2020): mask-weighted means of patch
    tokens over every row and every column of the patch grid (image-axis-aligned strips, no
    object-centred frame, no structural content), + learned strip position embeddings, the same
    2-layer transformer depth as ESA, mean-pooled."""

    def __init__(self, patch_dim: int, d: int = 192, heads: int = 4, depth: int = 2, max_len: int = 64):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, d))
        self.pos = nn.Parameter(torch.zeros(2, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.out_norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, patch, patch_hw, wfeat, blade, vein, patch_mask, return_attn=False):
        B, P, C = patch.shape
        h, w = patch_hw
        z = self.inp(patch).view(B, h, w, -1)
        m = patch_mask.view(B, h, w, 1).to(z.dtype)
        rows = (z * m).sum(2) / m.sum(2).clamp_min(1e-3)  # (B,h,d)
        cols = (z * m).sum(1) / m.sum(1).clamp_min(1e-3)  # (B,w,d)
        x = torch.cat([rows + self.pos[0, :h], cols + self.pos[1, :w]], 1)
        out = self.out_norm(self.enc(x).mean(1))
        return (out, {}) if return_attn else out
