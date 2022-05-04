# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from typing import NamedTuple
from scipy.sparse import linalg as sla


class EigenState(NamedTuple):
    """Object representing the energy, eigenvector and filling of an eigenstate."""

    energy: float = np.infty
    state: np.ndarray = None
    n_up: int = None
    n_dn: int = None


def compute_ground_state(basis, sham_func, *args, thresh=20, **kwargs):
    gs = EigenState()
    for sector in basis.iter_sectors():
        sham = sham_func(sector, *args, **kwargs)
        if sham.shape[0] == 1:
            # 1x1 matrix: eigenvalue problem trivial
            energy = sham[0, 0]
            state = np.array([1])
        elif 1 < sham.shape[0] <= thresh:
            # small matrix: solve full eigenvalue problem.
            energies, vectors = np.linalg.eigh(sham.toarray())
            idx = np.argmin(energies)
            energy, state = energies[idx], vectors[:, idx]
        else:
            # large matrix: use sparse implementation
            energies, vectors = sla.eigsh(sham, k=1, which="SA")
            energy, state = energies[0], vectors[:, 0]
        if energy < gs.energy:
            gs = EigenState(energy, state, sector.n_up, sector.n_dn)
    return gs


def solve_ground_state(basis, sham_func, *args, thresh=20, **kwargs):
    sector = basis.get_sector()
    sham = sham_func(sector, *args, **kwargs)
    if sham.shape[0] <= thresh:
        energies, vectors = np.linalg.eigh(sham.toarray())
        idx = np.argmin(energies)
        energy, state = energies[idx], vectors[:, idx]
    else:
        energies, vectors = sla.eigsh(sham, k=1, which="SA")
        energy, state = energies[0], vectors[:, 0]

    return EigenState(energy, state)
