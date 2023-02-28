# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2023, Dylan Jones

import numpy as np
from scipy.sparse import linalg as sla
from abc import ABC


class LinearOperator(sla.LinearOperator, ABC):
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
        ABC.__init__(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype: {self.dtype})"

    def toarray(self) -> np.ndarray:
        """Returns the `LinearOperator` in form of a dense array.

        This is a naive implementation for the sake of generality. Override for a more
        efficient implementation!
        """
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
            scaled.toarray = lambda: x * self.toarray()
        except AttributeError:
            pass
        return scaled

    def __rmul__(self, x):
        """Ensure methods in result."""
        scaled = super().__rmul__(x)
        try:
            scaled.trace = lambda: x * self.trace()
            scaled.toarray = lambda: x * self.toarray()
        except AttributeError:
            pass
        return scaled
