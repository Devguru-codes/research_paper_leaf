import numpy as np
import pytest
import torch
from skimage.measure import euler_number

from src.models.cubical_ect import (SweepGeometry, euler_increments, exact_euler, frame_coords,
                                    normalise_frame, pixel_coords, slab_statistics)

G = 64


def disk(r=20, c=(32, 32), n=G):
    yy, xx = np.mgrid[:n, :n]
    return ((yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r)


def shapes():
    d = disk()
    ann = disk(24) & ~disk(10)
    two = disk(10, (16, 16)) | disk(10, (46, 46))
    leaf = disk(25, (32, 26)) & disk(25, (32, 38))  # lens shape
    leaf_holes = leaf.copy()
    leaf_holes[30:33, 25:28] = False
    leaf_holes[30:33, 36:39] = False
    return {"disk": (d, 1), "annulus": (ann, 0), "two_disks": (two, 2), "lens": (leaf, 1),
            "lens_2holes": (leaf_holes, -1)}


def _run(binary: np.ndarray, geo, dtype=torch.float64):
    m = torch.as_tensor(binary, dtype=dtype)[None]
    c, s = normalise_frame(m)
    coords = frame_coords(pixel_coords(*m.shape[1:], dtype=dtype), c, s)
    return euler_increments(m, coords, geo)


@pytest.mark.parametrize("name", list(shapes()))
def test_exact_euler_matches_skimage(name):
    b, chi = shapes()[name]
    assert exact_euler(torch.as_tensor(b)[None]).item() == chi
    assert euler_number(b, connectivity=1) == chi


@pytest.mark.parametrize("name", list(shapes()))
def test_increments_telescope_to_total(name):
    b, chi = shapes()[name]
    geo = SweepGeometry(K=16, T=8).double()
    d = _run(b, geo)
    total = d.sum(-1)  # (1,K)
    assert torch.allclose(total, torch.full_like(total, float(chi)), atol=1e-9)


def test_slab_membership_partitions_unity():
    geo = SweepGeometry(K=8, T=16).double()
    h = torch.linspace(-5, 5, 1001, dtype=torch.float64)
    s = geo.slab_membership(h)
    assert torch.allclose(s.sum(-1), torch.ones_like(h), atol=1e-12)
    assert (s >= -1e-12).all()


def test_sharp_limit_matches_hard_sweep():
    """With large lambda the relaxed cumulative chi equals the exact chi of sublevel sets."""
    b, _ = shapes()["lens_2holes"]
    geo = SweepGeometry(K=8, T=8, kappa=400.0, nbins=4096).double()
    d = _run(b, geo)
    cum = d.cumsum(-1)[0]  # chi at t_1..t_T
    m = torch.as_tensor(b, dtype=torch.float64)[None]
    c, s = normalise_frame(m)
    coords = frame_coords(pixel_coords(G, G, dtype=torch.float64), c, s)[0]
    for k in range(geo.K):
        h = coords @ geo.dirs[k].double()
        for j, t in enumerate(geo.t_inner):
            # lower-star sublevel set of a vertex-filtration == binary set {h <= t} on vertices
            sub = (m[0] > 0) & (h <= t)
            exact = exact_euler(sub[None]).item()
            assert abs(cum[k, j].item() - exact) < 0.5, (k, j, cum[k, j].item(), exact)


def test_rotation_equivariance_unit_weights():
    """Rotating the binary blade by 90 deg cyclically shifts Delta chi by K/4 (exact)."""
    b, _ = shapes()["lens_2holes"]
    b = b & ~disk(6, (50, 20))  # break symmetry
    geo = SweepGeometry(K=32, T=16).double()
    d0 = _run(b, geo)
    for k in (1, 2, 3):
        dk = _run(np.rot90(b, k).copy(), geo)
        assert torch.allclose(dk, torch.roll(d0, shifts=k * geo.K // 4, dims=1), atol=1e-6), k


def test_soft_weights_monotone_cells():
    from src.models.cubical_ect import cubical_cells, vertex_heights
    a = torch.rand(2, 16, 16)
    h = vertex_heights(frame_coords(pixel_coords(16, 16), torch.zeros(2, 2), torch.ones(2)),
                       torch.eye(2))
    w, _ = cubical_cells(a, h)
    n = 16 * 16
    ne = 16 * 15
    wv, we1, we2, wq = w[:, :n], -w[:, n:n + ne], -w[:, n + ne:n + 2 * ne], w[:, n + 2 * ne:]
    av = a
    assert (we1.view(2, 16, 15) <= torch.minimum(av[:, :, :-1], av[:, :, 1:]) + 1e-7).all()
    assert (wq.view(2, 15, 15) <= we1.view(2, 16, 15)[:, :-1, :] + 1e-7).all()
    assert (wq.view(2, 15, 15) <= we2.view(2, 15, 16)[:, :, :-1] + 1e-7).all()


def test_gradients_flow_to_weights():
    a = torch.rand(2, 32, 32, requires_grad=True)
    geo = SweepGeometry(K=8, T=4)
    m = (torch.rand(2, 32, 32) > 0.3).float()
    c, s = normalise_frame(m)
    coords = frame_coords(pixel_coords(32, 32), c, s)
    d = euler_increments(a * m, coords, geo)
    st = slab_statistics(a * m, coords, geo)
    (d.square().sum() + st.sum()).backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


def _val(binary, geo, dtype=torch.float64):
    from src.models.cubical_ect import valuation_increments
    m = torch.as_tensor(binary, dtype=dtype)[None]
    c, s = normalise_frame(m)
    return valuation_increments(m, frame_coords(pixel_coords(*m.shape[1:], dtype=dtype), c, s), geo)


def test_hadwiger_totals_closed_form_square():
    """n x n pixel block: (n-1)^2 unit squares, perimeter 4(n-1), chi 1 -- for every direction."""
    n = 21
    b = np.zeros((G, G), bool)
    b[10:10 + n, 20:20 + n] = True
    tot = _val(b, SweepGeometry(K=8, T=8).double()).sum(2)[0]  # (K, 3)
    assert torch.allclose(tot, torch.tensor([(n - 1) ** 2, 4 * (n - 1), 1.0], dtype=torch.float64).expand(8, 3), atol=1e-9)


@pytest.mark.parametrize("name", list(shapes()))
def test_hadwiger_euler_matches_euler_increments(name):
    b, chi = shapes()[name]
    geo = SweepGeometry(K=16, T=8).double()
    v = _val(b, geo)
    assert torch.allclose(v[..., 2], _run(b, geo), atol=1e-9)
    assert torch.allclose(v[..., 2].sum(-1), torch.full((1, 16), float(chi), dtype=torch.float64), atol=1e-9)


def test_hadwiger_rotation_equivariance():
    b, _ = shapes()["lens_2holes"]
    b = b & ~disk(6, (50, 20))
    geo = SweepGeometry(K=32, T=16).double()
    v0 = _val(b, geo)
    for k in (1, 2, 3):
        assert torch.allclose(_val(np.rot90(b, k).copy(), geo), torch.roll(v0, k * geo.K // 4, dims=1), atol=1e-6)
