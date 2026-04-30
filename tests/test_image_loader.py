import tempfile
from pathlib import Path

import numpy as np
import pytest

from persviz.data import load_image_or_npy, synthetic_phantom


def test_load_npy_roundtrip(tmp_path):
    binary = synthetic_phantom(size=64, seed=0)
    p = tmp_path / "p.npy"
    np.save(p, binary)
    result = load_image_or_npy(p, target_size=64)
    assert result.shape == (64, 64)
    assert set(np.unique(result)).issubset({0, 1})


def test_load_npy_resizes(tmp_path):
    binary = synthetic_phantom(size=64, seed=0)
    p = tmp_path / "p.npy"
    np.save(p, binary)
    result = load_image_or_npy(p, target_size=128)
    assert result.shape == (128, 128)


def test_load_png(tmp_path):
    from PIL import Image
    binary = synthetic_phantom(size=64, seed=0)
    arr = (binary * 255).astype(np.uint8)
    p = tmp_path / "p.png"
    Image.fromarray(arr).save(p)
    result = load_image_or_npy(p, target_size=64)
    assert result.shape == (64, 64)
    assert set(np.unique(result)).issubset({0, 1})


def test_invert(tmp_path):
    binary = synthetic_phantom(size=64, seed=0)
    p = tmp_path / "p.npy"
    np.save(p, binary)
    a = load_image_or_npy(p, target_size=64, invert=False)
    b = load_image_or_npy(p, target_size=64, invert=True)
    assert (a + b == 1).all()


def test_constant_image_returns_zeros(tmp_path):
    arr = np.full((64, 64), 0.5)
    p = tmp_path / "p.npy"
    np.save(p, arr)
    result = load_image_or_npy(p, target_size=64)
    assert result.shape == (64, 64)
    assert (result == 0).all()
