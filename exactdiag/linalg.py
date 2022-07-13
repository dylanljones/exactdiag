# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from typing import NamedTuple
from scipy import linalg as la
from scipy.sparse import linalg as sla


class EigenState(NamedTuple):
    """Object representing the energy, eigenvector and filling of an eigenstate."""

    energy: float = np.infty
    state: np.ndarray = None
    n_up: int = None
    n_dn: int = None


def compute_ground_state(model, thresh=20):
    """Computes the ground state by iterating over the sectors of the Hamiltonian.

    Parameters
    ----------
    model : Model
        The model instance responsible for generating the Hamiltonian matrix. Also
        must contain the many body state basis of the system.
    thresh : int, optional
        The sector size threshold after which scipy's sparse linear algebra methods are
        used instead of the dense methods.

    Returns
    -------
    gs : EigenState
        The ground eigen state.
    """
    basis = model.basis
    gs = EigenState()
    for sector in basis.iter_sectors():
        sham = model.hamilton_operator(sector=sector)
        if sham.shape[0] == 1:
            # 1x1 matrix: eigenvalue problem trivial
            energy = sham.toarray()[0, 0]
            state = np.array([1])
        elif 1 < sham.shape[0] <= thresh:
            # small matrix: solve full eigenvalue problem.
            energies, vectors = la.eigh(sham.toarray())
            idx = np.argmin(energies)
            energy, state = energies[idx], vectors[:, idx]
        else:
            # large matrix: use sparse implementation
            energies, vectors = sla.eigsh(sham, k=1, which="SA")
            energy, state = energies[0], vectors[:, 0]
        if energy < gs.energy:
            gs = EigenState(energy, state, sector.n_up, sector.n_dn)
    return gs
