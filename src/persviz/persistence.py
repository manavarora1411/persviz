"""
Cubical persistent homology via the signed-EDT superlevel filtration.

We compute persistence of the superlevel filtration

    X_t = { x : phi(x) >= t },   t decreasing from +oo to -oo

on a 2D cubical complex (pixel grid with T-construction). The algorithm is
the standard boundary-matrix reduction of Edelsbrunner, Letscher &
Zomorodian (2002), "Topological Persistence and Simplification", Discrete
Comput. Geom. 28, 511-533, doi:10.1007/s00454-002-2885-2, as implemented
in GUDHI's CubicalComplex (Bauer et al. 2014).

GUDHI computes sublevel persistence by default. To convert superlevel
persistence on phi to sublevel persistence on (-phi):

    superlevel birth on phi at value b  <-->  sublevel birth on -phi at value -b
    superlevel death on phi at value d  <-->  sublevel death on -phi at value -d

with b > d (superlevel) corresponding to (-b) < (-d) (sublevel). We
negate the values back after computation so that the returned diagram
is in the original phi coordinates with birth > death.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gudhi import CubicalComplex


@dataclass
class PersistencePair:
    dim: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.birth - self.death


def superlevel_persistence(
    phi: np.ndarray,
    keep_infinite: bool = True,
) -> list[PersistencePair]:
    cc = CubicalComplex(top_dimensional_cells=-phi)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    raw = cc.persistence()
    pairs: list[PersistencePair] = []
    for dim, (b_neg, d_neg) in raw:
        b = -b_neg
        if d_neg == float("inf"):
            if not keep_infinite:
                continue
            d = -np.inf
        else:
            d = -d_neg
        pairs.append(PersistencePair(dim=dim, birth=float(b), death=float(d)))
    pairs.sort(key=lambda p: (p.dim, -p.persistence))
    return pairs


def betti_curves(
    pairs: list[PersistencePair],
    thresholds: np.ndarray,
    max_dim: int = 1,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for d in range(max_dim + 1):
        counts = np.zeros_like(thresholds, dtype=int)
        for p in pairs:
            if p.dim != d:
                continue
            alive = (thresholds <= p.birth) & (thresholds > p.death)
            counts += alive.astype(int)
        out[d] = counts
    return out
