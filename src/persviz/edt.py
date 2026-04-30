"""
Signed Euclidean distance transform.

Given a binary image with `1` = solid, `0` = void, the signed EDT is

    phi(x) = +d(x, solid)   if x is void
           = -d(x, void)    if x is solid

The interface phi = 0 is the solid-void boundary. Using superlevel sets
{phi >= t} sweeping t from +infinity to -infinity recovers the standard
filtration of Robins, Wood & Sheppard (2011), "Theory and algorithms for
constructing discrete Morse complexes from grayscale digital images",
IEEE PAMI 33(8), 1646-1658, doi:10.1109/TPAMI.2011.95.

The unsigned EDTs are computed via the separable O(n) algorithm of
Felzenszwalb & Huttenlocher (2012), as wrapped by SciPy.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_edt(binary: np.ndarray) -> np.ndarray:
    if binary.dtype != np.uint8 and binary.dtype != bool:
        binary = (binary > 0).astype(np.uint8)
    solid = binary.astype(bool)
    void = ~solid
    d_to_solid = distance_transform_edt(void)
    d_to_void = distance_transform_edt(solid)
    return d_to_solid - d_to_void
