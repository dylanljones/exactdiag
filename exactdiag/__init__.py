# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

from .utils import logger, CONFIG

from .basis import (
    UP,
    DN,
    SPIN_CHARS,
    state_label,
    binstr,
    binarr,
    binidx,
    overlap,
    occupations,
    create,
    annihilate,
    SpinState,
    State,
    Sector,
    Basis,
)

from .matrix import (
    matshow,
    cmatshow,
    transpose,
    hermitian,
    is_hermitian,
    diagonal,
    fill_diagonal,
    MatrixDecomposition,
    EigenDecomposition,
    QRDecomposition,
    SVDDecomposition,
)

from .op import LinearOperator
from .models.abc import ModelParameters, AbstractManyBodyModel
from .cache import AbstractCache, EigenCache, EigenCacheHDF5
from .linalg import EigenState, compute_ground_state

# from .operators import (
#     project_up,
#     project_dn,
#     project_elements_up,
#     project_elements_dn,
#     project_onsite_energy,
#     project_hubbard_inter,
#     project_hopping,
#     CreationOperator,
#     AnnihilationOperator,
#     HamiltonOperator,
# )
#
# from .greens import (
#     gf0_pole,
#     gf0_resolvent,
#     gf_lehmann,
#     compute_gf_lehmann,
#     gf_greater,
#     gf_lesser,
#     gf_tevo,
# )

from .disorder import (
    DisorderGenerator,
    DiagonalDisorder,
    SubstDiagonalDisorder,
    BinaryDiagonalDisorder,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
