# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2023, Dylan Jones

import bisect
import numpy as np
from scipy.sparse import linalg as sla, csr_matrix
from abc import ABC, abstractmethod
from .abc import ModelParameters
from ..operators import HamiltonOperator


class AbstractIsingModel(ModelParameters, ABC):
    """Abstract base class for Ising like models.

    The AbstractModel-class derives from ModelParameters.
    All parameters are accessable as attributes or dictionary-items.
    """

    def __init__(self, num_sites=0, **params):
        """Initializes the AbstractModel-instance with the given initial parameters.

        Parameters
        ----------
        **params: Initial parameters of the model.
        """
        ModelParameters.__init__(self, **params)
        self.nsites = 0
        self.basis = list()
        self.init_basis(num_sites)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({ModelParameters.__str__(self)})"

    def init_basis(self, num_sites):
        self.nsites = num_sites
        self.basis = list(np.arange(2**self.nsites))

    @property
    def nstates(self):
        return len(self.basis)

    @abstractmethod
    def _hamiltonian_data(self, states):
        pass

    def hamiltonian_data(self, states=None):
        if states is None:
            states = self.basis.copy()
        rows, cols, data = list(), list(), list()
        for row, col, val in self._hamiltonian_data(states):
            rows.append(row)
            cols.append(col)
            data.append(val)
        return data, np.array([rows, cols], dtype=np.int64)

    def hamilton_operator(self, states=None, dtype=None):
        size = len(self.basis) if states is None else len(states)
        data, indices = self.hamiltonian_data(states)
        return HamiltonOperator(size, data, indices, dtype=dtype)

    def hamiltonian(self, states=None, dtype=None):
        data, indices = self.hamiltonian_data(states)
        return csr_matrix((data, indices), dtype=dtype)


class IsingModel(AbstractIsingModel):
    def __init__(self, latt, jz=-1.0, hx=0.0, hz=0.0):
        super().__init__(latt.num_sites, jz=jz, hx=hx, hz=hz)
        self.latt = latt

    def _hamiltonian_data(self, states):
        for i, state in enumerate(states):
            for site1 in range(self.nsites):
                op1 = 1 << site1
                # sz.sz
                if self.jz:
                    val = 0
                    for site2 in self.latt.nearest_neighbors(site1, unique=True):
                        op2 = 1 << site2
                        if bool(op1 & state) == bool(op2 & state):
                            val += self.jz
                        else:
                            val -= self.jz
                    yield i, i, val
                # sz
                if self.hz:
                    val = -self.hz if bool(op1 & state) else self.hz
                    yield i, i, val
                # sx
                if self.hx:
                    # index of state with flipped bit
                    new_state = state ^ op1
                    # j = np.searchsorted(states, new_state)
                    j = bisect.bisect_left(states, new_state)
                    yield i, j, self.hx

    def ground_state(self, thresh=10):
        ham = self.hamilton_operator()
        if self.nsites >= thresh:
            return sla.eigsh(ham, k=1)  # noqa
        else:
            energies, vectors = np.linalg.eigh(ham.toarray())
            idx = np.argmin(energies)
            return energies[idx], vectors[:, idx]


def compute_magnetization(psi):
    r"""Computes the magnetization `M` of the state `ψ`.

    The magnetization .math:`M = ⟨ψ|\hat{M}|ψ⟩`is the expectation value of the operator
    .. math::

        \hat{M} = \frac{1}{N} \sum_{i=1}^{N} σ^z_i

    Parameters
    ----------
    psi : (N, ) np.ndarray
        The state used to compute the magnetization.

    Returns
    -------
    mag : float
        The magentization of the system in the state.
    """
    magnetization = 0
    nsites = int(np.log2(len(psi)))
    psi2 = np.square(psi)
    for i in range(len(psi)):
        state = i
        mag = 0
        for site in range(nsites):
            op = 1 << site
            s = 1 if bool(state & op) else -1
            mag += s * psi2[i]
        mag = abs(mag) / nsites
        assert -1e-10 <= mag <= 1.0 + 1e-10
        magnetization += max(0.0, min(mag, 1.0))
    return magnetization
