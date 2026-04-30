import numpy as np
import pytest

from persviz.data import synthetic_phantom
from persviz.edt import signed_edt
from persviz.pairing import paired_features
from persviz.persistence import betti_curves, superlevel_persistence


def _annulus(N=64, r_in=8, r_out=18):
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    r = np.hypot(yy - N / 2, xx - N / 2)
    return ((r > r_in) & (r < r_out)).astype(np.uint8)


def test_essential_h0_count():
    binary = _annulus()
    phi = signed_edt(binary)
    pairs = superlevel_persistence(phi)
    essential = [p for p in pairs if p.dim == 0 and not np.isfinite(p.death)]
    assert len(essential) == 1


def test_significant_h1_count_for_annulus():
    binary = _annulus()
    phi = signed_edt(binary)
    pairs = superlevel_persistence(phi)
    h1 = [
        p for p in pairs
        if p.dim == 1 and np.isfinite(p.death) and p.persistence > 1.0
    ]
    assert len(h1) >= 1


def test_birth_greater_than_death():
    binary = synthetic_phantom(size=64, seed=0)
    phi = signed_edt(binary)
    pairs = superlevel_persistence(phi)
    for p in pairs:
        if np.isfinite(p.death):
            assert p.birth >= p.death


def test_paired_features_match_persistence():
    binary = synthetic_phantom(size=64, seed=1)
    phi = signed_edt(binary)
    pairs = superlevel_persistence(phi)
    feats = paired_features(phi)
    pair_set = sorted(
        [(p.dim, round(p.birth, 4), round(p.death, 4) if np.isfinite(p.death) else None)
         for p in pairs]
    )
    feat_set = sorted(
        [(f.dim, round(f.birth, 4), round(f.death, 4) if np.isfinite(f.death) else None)
         for f in feats]
    )
    assert pair_set == feat_set


def test_betti_curve_essential_component():
    binary = synthetic_phantom(size=64, seed=2)
    phi = signed_edt(binary)
    pairs = superlevel_persistence(phi)
    ts = np.linspace(phi.max() + 0.5, phi.min() - 0.5, 30)
    bc = betti_curves(pairs, ts, max_dim=1)
    assert bc[0][-1] >= 1
    assert bc[0][0] == 0


def test_paired_feature_cells_inside_image():
    binary = synthetic_phantom(size=48, seed=3)
    phi = signed_edt(binary)
    feats = paired_features(phi)
    h, w = phi.shape
    for f in feats:
        br, bc = f.birth_cell
        assert 0 <= br < h and 0 <= bc < w
        if f.death_cell is not None:
            dr, dc = f.death_cell
            assert 0 <= dr < h and 0 <= dc < w
