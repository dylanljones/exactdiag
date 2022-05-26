# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

from .utils import logger

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
    transpose,
    hermitian,
    is_hermitian,
    diagonal,
    fill_diagonal,
    MatrixDecomposition,
    EigenDecomposition,
    QRDecomposition,
    SVDDecomposition,
    EigenState,
)

from .operators import (
    project_up,
    project_dn,
    project_elements_up,
    project_elements_dn,
    project_onsite_energy,
    project_hubbard_inter,
    project_hopping,
    LinearOperator,
    CreationOperator,
    AnnihilationOperator,
    HamiltonOperator,
)
from exactdiag.models.abc import ModelParameters, AbstractManyBodyModel
from .greens import gf0_pole, gf_lehmann, gf_greater, gf_lesser, gf_tevo

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
