# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

"""This module contains tools for working with linear operators in sparse format."""

import abc
import numpy as np
from bisect import bisect_left
import scipy.sparse.linalg as sla
from .basis import UP, SPIN_CHARS
from .project import project_up, project_dn

__all__ = [
    "LinearOperator",
    "CreationOperator",
    "AnnihilationOperator",
    "HamiltonOperator",
]


class LinearOperator(sla.LinearOperator, abc.ABC):
    """Abstract base class for linear operators.

    Turns any class that imlements the `_matvec`- or `_matmat`-method and
    turns it into an object that behaves like a linear operator.

    Abstract Methods
    ----------------
    _matvec(v): Matrix-vector multiplication.
        Performs the operation y=A*v where A is an MxN linear operator and
        v is a column vector or 1-d array.
        Implementing _matvec automatically implements _matmat (using a naive algorithm).

    _matmat(X): Matrix-matrix multiplication.
        Performs the operation Y=A*X where A is an MxN linear operator and
        X is a NxM matrix.
        Implementing _matmat automatically implements _matvec (using a naive algorithm).

    _adjoint(): Hermitian adjoint.
        Returns the Hermitian adjoint of self, aka the Hermitian conjugate or Hermitian
        transpose. For a complex matrix, the Hermitian adjoint is equal to the conjugate
        transpose. Can be abbreviated self.H instead of self.adjoint(). As with
        _matvec and _matmat, implementing either _rmatvec or _adjoint implements the
        other automatically. Implementing _adjoint is preferable!

    _trace(): Trace of operator.
        Computes the trace of the operator using the a dense array.
        Implementing _trace with a more sophisticated method is preferable!
    """

    def __init__(self, shape, dtype=None):
        sla.LinearOperator.__init__(self, shape=shape, dtype=dtype)
        abc.ABC.__init__(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype: {self.dtype})"

    def array(self) -> np.ndarray:
        """Returns the `LinearOperator` in form of a dense array."""
        x = np.eye(self.shape[1], dtype=self.dtype)
        return self.matmat(x)

    def _trace(self) -> float:
        """Naive implementation of trace. Override for more efficient calculation."""
        x = np.eye(self.shape[1], dtype=self.dtype)
        return float(np.trace(self.matmat(x)))

    def trace(self) -> float:
        """Computes the trace of the ``LinearOperator``."""
        return self._trace()

    def __mul__(self, x):
        """Ensure methods in result."""
        scaled = super().__mul__(x)
        try:
            scaled.trace = lambda: x * self.trace()
            scaled.array = lambda: x * self.array()
        except AttributeError:
            pass
        return scaled

    def __rmul__(self, x):
        """Ensure methods in result."""
        scaled = super().__rmul__(x)
        try:
            scaled.trace = lambda: x * self.trace()
            scaled.array = lambda: x * self.array()
        except AttributeError:
            pass
        return scaled


# -- Creation- and Annihilation-Operators ----------------------------------------------


class CreationOperator(LinearOperator):
    """Fermionic creation operator as LinearOperator."""

    def __init__(self, sector, sector_p1, pos=0, sigma=UP):
        dim_origin = sector.size
        if sigma == UP:
            dim_target = sector_p1.num_up * sector.num_dn
        else:
            dim_target = sector_p1.num_dn * sector.num_up

        super().__init__(shape=(dim_target, dim_origin), dtype=np.complex64)
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_p1 = sector_p1

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, matvec, x):
        op = 1 << self.pos
        num_dn = len(self.sector.dn_states)
        all_dn = np.arange(num_dn)
        for up_idx, up in enumerate(self.sector.up_states):
            if not (up & op):
                new = up ^ op
                idx_new = bisect_left(self.sector_p1.up_states, new)
                origins = project_up(up_idx, num_dn, all_dn)
                targets = project_up(idx_new, num_dn, all_dn)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _build_dn(self, matvec, x):
        op = 1 << self.pos
        num_dn = self.sector.num_dn
        all_up = np.arange(self.sector.num_up)
        for dn_idx, dn in enumerate(self.sector.dn_states):
            if not (dn & op):
                new = dn ^ op
                idx_new = bisect_left(self.sector_p1.dn_states, new)
                origins = project_dn(dn_idx, num_dn, all_up)
                targets = project_dn(idx_new, num_dn, all_up)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _matvec(self, x):
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec

    def _adjoint(self):
        return AnnihilationOperator(self.sector_p1, self.sector, self.pos, self.sigma)


class AnnihilationOperator(LinearOperator):
    """Fermionic annihilation operator as LinearOperator."""

    def __init__(self, sector, sector_m1, pos=0, sigma=UP):
        dim_origin = sector.size
        if sigma == UP:
            dim_target = sector_m1.num_up * sector.num_dn
        else:
            dim_target = sector_m1.num_dn * sector.num_up

        super().__init__(shape=(dim_target, dim_origin), dtype=np.complex64)
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_m1 = sector_m1

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, matvec, x):
        op = 1 << self.pos
        num_dn = len(self.sector.dn_states)
        all_dn = np.arange(num_dn)
        for up_idx, up in enumerate(self.sector.up_states):
            if up & op:
                new = up ^ op
                idx_new = bisect_left(self.sector_m1.up_states, new)
                origins = project_up(up_idx, num_dn, all_dn)
                targets = project_up(idx_new, num_dn, all_dn)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _build_dn(self, matvec, x):
        op = 1 << self.pos
        num_dn = self.sector.num_dn
        all_up = np.arange(self.sector.num_up)
        for dn_idx, dn in enumerate(self.sector.dn_states):
            if dn & op:
                new = dn ^ op
                idx_new = bisect_left(self.sector_m1.dn_states, new)
                origins = project_dn(dn_idx, num_dn, all_up)
                targets = project_dn(idx_new, num_dn, all_up)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _matvec(self, x):
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec

    def _adjoint(self):
        return CreationOperator(self.sector_m1, self.sector, self.pos, self.sigma)


class HamiltonOperator(LinearOperator):
    """Hamiltonian as LinearOperator."""

    def __init__(self, size, data, indices, dtype=None):
        data = np.asarray(data)
        indices = np.asarray(indices)
        if dtype is None:
            dtype = data.dtype
        super().__init__((size, size), dtype=dtype)
        self.data = data
        self.indices = indices.T

    def _matvec(self, x) -> np.ndarray:
        matvec = np.zeros_like(x)
        for (row, col), val in zip(self.indices, self.data):
            matvec[col] += val * x[row]
        return matvec

    def _adjoint(self) -> "HamiltonOperator":
        """Hamiltonian is hermitian."""
        return self

    def _trace(self) -> float:
        """More efficient trace."""
        # Check elements where the row equals the column
        indices = np.where(self.indices[:, 0] == self.indices[:, 1])[0]
        # Return sum of diagonal elements
        return float(np.sum(self.data[indices]))
