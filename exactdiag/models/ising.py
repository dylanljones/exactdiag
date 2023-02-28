# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2023, Dylan Jones

import numpy as np
from scipy.sparse import csr_matrix
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
