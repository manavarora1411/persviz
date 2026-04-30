"""persviz: persistent homology visualization for porous media."""

from persviz.data import load_binary_npy, load_image_or_npy, synthetic_phantom
from persviz.edt import signed_edt
from persviz.pairing import PairedFeature, paired_features
from persviz.persistence import (
    PersistencePair,
    betti_curves,
    superlevel_persistence,
)

__version__ = "0.1.0"

__all__ = [
    "PairedFeature",
    "PersistencePair",
    "betti_curves",
    "load_binary_npy",
    "load_image_or_npy",
    "paired_features",
    "signed_edt",
    "superlevel_persistence",
    "synthetic_phantom",
]
