"""
Birth-death pairing with geometric cell coordinates.

For each persistence pair (b, d), we recover representative top-dimensional
cells in the cubical complex - one whose value caused the birth, one whose
value caused the death. This lets the animation draw, at the moment a
feature dies, a visual link from the killing cell back to the cell that
gave birth to it - the geometric realization of the pairing produced by
ELZ matrix reduction.

Implementation notes
--------------------
GUDHI's `cofaces_of_persistence_pairs()` returns top-cell linear indices
into the flattened input. With T-construction (top-dimensional cells
given), these indices are representatives of the birth and death events -
not necessarily the global argmin/argmax. We use `persistence()` for the
canonical (birth, death) values and `cofaces_of_persistence_pairs()`
purely for cell coordinates. We match the two outputs by sorting both
lists per dimension by birth value, since GUDHI does not guarantee
identical iteration order between the two API calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gudhi import CubicalComplex


@dataclass
class PairedFeature:
    dim: int
    birth: float
    death: float
    birth_cell: tuple[int, int]
    death_cell: tuple[int, int] | None

    @property
    def persistence(self) -> float:
        return self.birth - self.death


def paired_features(phi: np.ndarray) -> list[PairedFeature]:
    neg = -phi
    cc = CubicalComplex(top_dimensional_cells=neg)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    pers_raw = cc.persistence()
    regular, essential = cc.cofaces_of_persistence_pairs()
    shape = phi.shape

    pers_finite: dict[int, list[tuple[float, float]]] = {}
    pers_essential: list[tuple[int, float]] = []
    for dim, (b_neg, d_neg) in pers_raw:
        b = -b_neg
        if d_neg == float("inf"):
            pers_essential.append((dim, b))
        else:
            pers_finite.setdefault(dim, []).append((b, -d_neg))

    coface_finite: dict[int, list[tuple[int, int, float, float]]] = {}
    for dim, arr in enumerate(regular):
        if len(arr) == 0:
            continue
        items: list[tuple[int, int, float, float]] = []
        for b_idx, d_idx in arr:
            b_idx, d_idx = int(b_idx), int(d_idx)
            b_val = float(neg.flat[b_idx])
            d_val = float(neg.flat[d_idx])
            items.append((b_idx, d_idx, -b_val, -d_val))
        coface_finite[dim] = items

    coface_essential: list[int] = []
    if essential and len(essential[0]) > 0:
        coface_essential = [int(i) for i in essential[0]]

    out: list[PairedFeature] = []
    for dim, pairs in pers_finite.items():
        cells = coface_finite.get(dim, [])
        pairs_sorted = sorted(enumerate(pairs), key=lambda x: (x[1][0], x[1][1]))
        cells_sorted = sorted(cells, key=lambda x: (x[2], x[3]))
        if len(pairs_sorted) != len(cells_sorted):
            n = min(len(pairs_sorted), len(cells_sorted))
            pairs_sorted = pairs_sorted[:n]
            cells_sorted = cells_sorted[:n]
        for (_orig_idx, (b, d)), (b_idx, d_idx, _bv, _dv) in zip(pairs_sorted, cells_sorted):
            br, bc = np.unravel_index(b_idx, shape)
            dr, dc = np.unravel_index(d_idx, shape)
            out.append(
                PairedFeature(
                    dim=dim,
                    birth=float(b),
                    death=float(d),
                    birth_cell=(int(br), int(bc)),
                    death_cell=(int(dr), int(dc)),
                )
            )

    if len(pers_essential) == len(coface_essential):
        ess_sorted_pers = sorted(pers_essential, key=lambda x: x[1])
        ess_sorted_cells = sorted(coface_essential, key=lambda i: float(neg.flat[i]))
        for (dim, b), idx in zip(ess_sorted_pers, ess_sorted_cells):
            br, bc = np.unravel_index(idx, shape)
            out.append(
                PairedFeature(
                    dim=dim,
                    birth=float(b),
                    death=float("-inf"),
                    birth_cell=(int(br), int(bc)),
                    death_cell=None,
                )
            )
    else:
        argmax = np.unravel_index(int(np.argmax(phi)), shape)
        for dim, b in pers_essential:
            out.append(
                PairedFeature(
                    dim=dim,
                    birth=float(b),
                    death=float("-inf"),
                    birth_cell=(int(argmax[0]), int(argmax[1])),
                    death_cell=None,
                )
            )

    out.sort(key=lambda p: (p.dim, -p.persistence))
    return out
