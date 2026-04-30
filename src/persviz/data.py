"""
Synthetic porous-media phantom generator and .npy loader.

The phantom mimics a 2D slice of a granular porous medium: randomly placed
Gaussian "grains" superimposed and thresholded. This is the standard
phantom model used in the Robins-Wood-Sheppard (2011) line of work on
sandstone microstructure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def synthetic_phantom(
    size: int = 128,
    n_grains: int = 35,
    grain_radius_range: tuple[float, float] = (6.0, 14.0),
    threshold: float = 0.5,
    seed: int | None = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    field = np.zeros((size, size), dtype=np.float64)
    for _ in range(n_grains):
        cy = rng.uniform(0, size)
        cx = rng.uniform(0, size)
        r = rng.uniform(*grain_radius_range)
        field += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    field /= field.max()
    return (field > threshold).astype(np.uint8)


def load_binary_npy(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8)
    return arr


def load_image_or_npy(
    path: str | Path,
    target_size: int = 128,
    invert: bool = False,
) -> np.ndarray:
    from pathlib import Path as _P
    suffix = _P(path).suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
    else:
        from skimage.io import imread
        from skimage.color import rgb2gray, rgba2rgb
        img = imread(path)
        if img.ndim == 3:
            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            img = rgb2gray(img)
        arr = img.astype(float)

    from skimage.transform import resize
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        arr = resize(arr.astype(float), (target_size, target_size), anti_aliasing=True)

    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)

    from skimage.filters import threshold_otsu
    if arr.std() < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    thresh = threshold_otsu(arr)
    binary = (arr > thresh).astype(np.uint8)
    if invert:
        binary = 1 - binary
    return binary
