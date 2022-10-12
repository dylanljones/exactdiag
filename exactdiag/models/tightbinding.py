# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np


def tight_binding_hamiltonian(latt, eps=0.0, hop=1.0, dtype=None):
    """Constructs a tight-binding Hamiltonian for a lattice model.

    Parameters
    ----------
    latt : (N, ) Lattice
        The lattice structure of the model.
    eps : float or (M, M) np.ndarray
        The on-site energy of the model. Can be a scalar in the case of one orbital
        (M=1) or a matrix for multiple orbitals (M>1).
    hop : float or (M, M) np.ndarray
        The hopping energy of the model. Can be a scalar in the case of one orbital
        (M=1) or a matrix for multiple orbitals (M>1).
    dtype : str of np.dtype
        The data-type of the resulting Hamiltonian matrix.

    Returns
    -------
    ham : (N*M, N*M) np.ndarray
        The tight-binding Hamiltonian in matrix representation.
    """
    eps = np.asanyarray(eps)
    hop = np.asanyarray(hop)
    norbs = 1 if len(eps.shape) < 2 else eps.shape[-1]
    norbs2 = 1 if len(hop.shape) < 2 else hop.shape[-1]
    if norbs != norbs2:
        raise ValueError(
            f"Size of on-site energy {norbs} does not match "
            f"size of hopping energy {norbs2}!"
        )
    if dtype is None:
        # Default dtype matches dtype of arguments
        dtype = (eps + hop).dtype

    dmap = latt.dmap()
    if norbs == 1:
        # Build single orbital Hamiltonian
        data = np.zeros(dmap.size, dtype=dtype)
        data[dmap.onsite()] = eps
        data[dmap.hopping()] = hop
        hams = dmap.build_csr(data)
    else:
        # Build ulti-.
        data = np.zeros((dmap.size, norbs, norbs), dtype=dtype)
        data[dmap.onsite()] = eps
        data[dmap.hopping()] = hop
        hams = dmap.build_bsr(data)

    return hams.toarray()
