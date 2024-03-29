# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones, Andreas Weh

"""This module contains tools for working with linear operators in sparse format.

The ordering of the creation/annihilation operators of basis states uses the following
convention: First, all spin-up creation operators are filled up from left to right
with increasing site index. Then, the spin-down operators are added, e.g.:
.. math::
    |ψ⟩ = c^†_{0,↑} ... c^†_{N-1,↑} c^†_{0,↓} ... c^†_{N-1,↓} |0⟩
    ⟨ψ| = ⟨0| c_{N-1,↓} ... c_{0,↓} c_{N-1,↑} ... c_{0,↑}

The ordering of the operators is important for the computation of signs in the
hopping terms of Hamilton operators and the matrix representation of the
creation/annihilation operators.
"""

import hashlib
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from .basis import UP, SPIN_CHARS
from .op import LinearOperator


__all__ = [
    "project_up",
    "project_dn",
    "project_elements_up",
    "project_elements_dn",
    "project_hubbard_inter",
    "project_onsite_energy",
    "project_hopping",
    "CreationOperator",
    "AnnihilationOperator",
    "HamiltonOperator",
    "bisect_left",
    "bit_count",
    "bit_count_between",
]

_jitkw = dict(fastmath=True, nogil=True)


def project_up(up_idx, num_dn_states, dn_indices):
    """Projects a spin-up state onto the full basis(-sector).

    Parameters
    ----------
    up_idx : int
        The index of the up-state to project.
    num_dn_states : int
        The number N of spin-down states of the basis(-sector).
    dn_indices : int or (N, ) np.ndarray
        An array of the indices of all spin-down states in the basis(-sector).

    Returns
    -------
    projected_up : np.ndarray
        The indices of the projected up-state.

    Examples
    --------
    >>> num_dn = 4
    >>> all_dn = np.arange(4)
    >>> project_up(0, num_dn, all_dn)
    array([0, 1, 2, 3])

    >>> project_up(1, num_dn, all_dn)
    array([4, 5, 6, 7])
    """
    return np.atleast_1d(up_idx * num_dn_states + dn_indices)


def project_dn(dn_idx, num_dn_states, up_indices):
    """Projects a spin-down state onto the full basis(-sector).

    Parameters
    ----------
    dn_idx : int
        The index of the down-state to project.
    num_dn_states : int
        The number N of spin-down states of the basis(-sector).
    up_indices : int or (N, ) np.ndarray
        An array of the indices of all spin-up states in the basis(-sector).

    Returns
    -------
    projected_dn : np.ndarray
        The indices of the projected down-state.

    Examples
    --------
    >>> num_dn = 4
    >>> all_up = np.arange(4)
    >>> project_dn(0, num_dn, all_up)
    array([ 0,  4,  8, 12])

    >>> project_dn(1, num_dn, all_up)
    array([ 1,  5,  9, 13])
    """
    return np.atleast_1d(up_indices * num_dn_states + dn_idx)


@njit(**_jitkw)
def project_elements_up(up_idx, num_dn_states, dn_indices, value, target=None):
    """Projects a value for a spin-up state onto the full basis(-sector).

    Parameters
    ----------
    up_idx : int
        The index of the up-state to project.
    num_dn_states : int
        The total number of spin-down states of the basis(-sector).
    dn_indices : int or np.ndarray
        An array of the indices of all spin-down states in the basis(-sector).
    value : float or complex
        The value to project.
    target : int or np.ndarray, optional
        The target index/indices for the projection. This is only needed
        for non-diagonal elements.

    Yields
    -------
    row : int
        The row-index of the element.
    col : int
        The column-index of the element.
    value : float or complex
        The value of the matrix-element.

    Examples
    --------
    >>> num_dn = 4
    >>> all_dn = np.arange(4)
    >>> np.array(list(project_elements_up(0, num_dn, all_dn, value=1)))
    array([[0, 0, 1],
           [1, 1, 1],
           [2, 2, 1],
           [3, 3, 1]])

    >>> np.array(list(project_elements_up(0, num_dn, all_dn, value=1, target=1)))
    array([[0, 4, 1],
           [1, 5, 1],
           [2, 6, 1],
           [3, 7, 1]])

    >>> np.array(list(project_elements_up(1, num_dn, all_dn, value=1)))
    array([[4, 4, 1],
           [5, 5, 1],
           [6, 6, 1],
           [7, 7, 1]])

    >>> np.array(list(project_elements_up(1, num_dn, all_dn, value=1, target=2)))
    array([[ 4,  8,  1],
           [ 5,  9,  1],
           [ 6, 10,  1],
           [ 7, 11,  1]])
    """
    origins = np.atleast_1d(up_idx * num_dn_states + dn_indices)
    if target is None:
        targets = origins
    else:
        targets = np.atleast_1d(target * num_dn_states + dn_indices)

    for row, col in zip(origins, targets):
        yield row, col, value


@njit(**_jitkw)
def project_elements_dn(dn_idx, num_dn_states, up_indices, value, target=None):
    """Projects a value for a spin-down state onto the full basis(-sector).

    Parameters
    ----------
    dn_idx : int
        The index of the down-state to project.
    num_dn_states : int
        The total number of spin-down states of the basis(-sector).
    up_indices : int or np.ndarray
        An array of the indices of all spin-up states in the basis(-sector).
    value : float or complex
        The value to project.
    target : int or np.ndarray, optional
        The target index/indices for the projection. This is only needed
        for non-diagonal elements.

    Yields
    -------
    row : int
        The row-index of the element.
    col : int
        The column-index of the element.
    value : float or complex
        The value of the matrix-element.

    Examples
    --------
    >>> num_dn = 4
    >>> all_up = np.arange(4)
    >>> np.array(list(project_elements_dn(0, num_dn, all_up, value=1)))
    array([[ 0,  0,  1],
           [ 4,  4,  1],
           [ 8,  8,  1],
           [12, 12,  1]])

    >>> np.array(list(project_elements_dn(0, num_dn, all_up, value=1, target=1)))
    array([[ 0,  1,  1],
           [ 4,  5,  1],
           [ 8,  9,  1],
           [12, 13,  1]])

    >>> np.array(list(project_elements_dn(1, num_dn, all_up, value=1)))
    array([[ 1,  1,  1],
           [ 5,  5,  1],
           [ 9,  9,  1],
           [13, 13,  1]])

    >>> np.array(list(project_elements_dn(1, num_dn, all_up, value=1, target=2)))
    array([[ 1,  2,  1],
           [ 5,  6,  1],
           [ 9, 10,  1],
           [13, 14,  1]])
    """
    origins = np.atleast_1d(up_indices * num_dn_states + dn_idx)
    if target is None:
        targets = origins
    else:
        targets = np.atleast_1d(up_indices * num_dn_states + target)

    for row, col in zip(origins, targets):
        yield row, col, value


# -- Helper methods --------------------------------------------------------------------


@njit("float64(int32, float64[:])", cache=True, **_jitkw)
def weighted_element(state, values):
    """Computes the value of a specific matrix element of a many-body Hamiltonian.

    Maps the values to the state of each site and computes the sum. This is equivalent
    of multiplying the value array with the bit-values of a state number.

    Parameters
    ----------
    state : int
        The number representing the binary state.
    values : (N, ) float np.ndarray
        The values of the Hamiltonian of an N site lattice model. This can be, for
        example, an array filled with the on-site energies or the interaction energies.

    Returns
    -------
    value : float
        The weighted values.
    """
    value = 0.0
    for i in range(values.shape[0]):
        if state & (1 << i):
            value += values[i]
    return value


@njit("int32(int32, int32)", cache=True, **_jitkw)
def bit_count(number, width):
    """Counts the number of bits with value 1.

    Parameters
    ----------
    number : int
        The number representing the binary state.
    width : int
        Number N of digits used.

    Returns
    -------
    count : int
        The number of bits set to 1.
    """
    count = 0
    for i in range(width):
        if bool(number & (1 << i)):
            count += 1
    return count


@njit("int32(int32, int32, int32)", cache=True, **_jitkw)
def bit_count_between(number, start, stop):
    count = 0
    for i in range(start, stop):
        if bool(number & (1 << i)):
            count += 1
    return count


@njit("int32(int32[:], int32)", cache=True, **_jitkw)
def bisect_left(a, x):
    """Locate the insertion point for x in `a` to maintain a sorted order.

    Parameters
    ----------
    a : (N, ) int np.ndarray
        The input array.
    x : int
        The value to insert into the input array.

    Returns
    -------
    i : int
        The insertion point of `x` in `a`.
    """
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# -- Hamiltonian projectors ------------------------------------------------------------


@njit("(int32[:], int32[:], float64[:])", **_jitkw)
def project_hubbard_inter(up_states, dn_states, u):
    """Projects the on-site interaction of a many-body Hamiltonian onto the full basis.

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    u : array_like
        The on-site interaction.

    Yields
    ------
    row : int
        The row-index of the on-site interaction.
    col : int
        The column-index of the on-site interaction.
    value : float
        The on-site interaction.

    Examples
    --------
    >>> from exactdiag import Basis
    >>> basis = Basis(num_sites=2)
    >>> sector = basis.get_sector(n_up=1, n_dn=1)
    >>> states_up, states_dn = sector.up_states, sector.dn_states
    >>> inter = [1.0, 2.0]
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in project_hubbard_inter(states_up, states_dn, inter):
    ...     ham[i, j] += val
    >>> ham
    array([[1., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 2.]])

    >>> from exactdiag import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)
    """
    # num_sites = len(u)
    num_dn = len(dn_states)

    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            # weights = binarray(up & dn, num_sites)
            # energy = np.sum(u * weights
            energy = weighted_element(up & dn, u)
            if energy:
                origin = up_idx * num_dn + dn_idx
                yield origin, origin, energy


@njit("(int32[:], int32[:], float64[:])", **_jitkw)
def project_onsite_energy(up_states, dn_states, eps):
    """Projects the on-site energy of a many-body Hamiltonian onto full basis(-sector).

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    eps : array_like
        The on-site energy.

    Yields
    ------
    row : int
        The row-index of the on-site energy.
    col : int
        The column-index of the on-site energy.
    value : float
        The on-site energy.

    Examples
    --------
    >>> from exactdiag import Basis
    >>> basis = Basis(num_sites=2)
    >>> sector = basis.get_sector(n_up=1, n_dn=1)
    >>> states_up, states_dn = sector.up_states, sector.dn_states
    >>> energies = [1.0, 2.0]
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in project_onsite_energy(states_up, states_dn, energies):
    ...     ham[i, j] += val
    >>> ham
    array([[2., 0., 0., 0.],
           [0., 3., 0., 0.],
           [0., 0., 3., 0.],
           [0., 0., 0., 4.]])

    >>> from exactdiag import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)
    """
    # num_sites = len(eps)
    num_dn = len(dn_states)
    all_up = np.arange(len(up_states), dtype=np.int32) * num_dn
    all_dn = np.arange(num_dn, dtype=np.int32)

    # Spin-up elements
    for up_idx, up in enumerate(up_states):
        # weights = binarray(up, num_sites)
        # energy = np.sum(eps * weights)
        energy = weighted_element(up, eps)
        if energy:
            origins = up_idx * num_dn + all_dn
            for origin in origins:
                yield origin, origin, energy

    # Spin-dn elements
    for dn_idx, dn in enumerate(dn_states):
        # weights = binarray(dn, num_sites)
        # energy = np.sum(eps * weights)
        energy = weighted_element(dn, eps)
        if energy:
            origins = all_up + dn_idx
            for origin in origins:
                yield origin, origin, energy


@njit("(int32[:], int32, int32, float64)", **_jitkw)
def _compute_hopping_term(states, site1, site2, hop):
    assert site1 < site2

    for i, ini in enumerate(states):
        op1 = 1 << site1  # Selects bit with index `site1`
        occ1 = ini & op1  # Value of bit of state with index `site1`
        tmp = ini ^ op1  # Annihilate/create electron at `site1`

        op2 = 1 << site2  # Selects bit with index `site2`
        occ2 = ini & op2  # Value of bit of state with index `site2`
        new = tmp ^ op2  # Create/annihilate electron at `site1`

        if occ1 and not occ2:
            # Hopping from `site1` to `site2` possible
            sign = (-1) ** bit_count_between(ini, site1 + 1, site2)
            j = bisect_left(states, new)
            yield i, j, sign * hop

        elif occ2 and not occ1:
            # Hopping from `site2` to `site1` possible
            sign = (-1) ** bit_count_between(ini, site1 + 1, site2)
            j = bisect_left(states, new)
            yield i, j, sign * hop


@njit("(int32[:], int32[:], int32, int32, float64)", **_jitkw)
def project_hopping(up_states, dn_states, site1, site2, hop):
    """Projects the hopping between two sites onto full basis.

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    site1 : int
        The first site of the hopping pair. This has to be the lower index of the two
        sites.
    site2 : int
        The second site of the hopping pair. This has to be the larger index of the two
        sites.
    hop : float, optional
        The hopping energy between the two sites.

    Yields
    ------
    row : int
        The row-index of the hopping element.
    col : int
        The column-index of the hopping element.
    value : float
        The hopping energy.

    Examples
    --------
    >>> from exactdiag import Basis
    >>> basis = Basis(num_sites=2)
    >>> sector = basis.get_sector(n_up=1, n_dn=1)
    >>> nsites = sector.num_sites
    >>> up, dn = sector.up_states, sector.dn_states
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for r, c, val in project_hopping(up, dn, site1=0, site2=1, hop=1.0):
    ...     ham[r, c] += val
    >>> ham
    array([[0., 1., 1., 0.],
           [1., 0., 0., 1.],
           [1., 0., 0., 1.],
           [0., 1., 1., 0.]])

    >>> from exactdiag import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)
    """
    num_dn = len(dn_states)
    all_up = np.arange(len(up_states), dtype=np.int32) * num_dn
    all_dn = np.arange(num_dn, dtype=np.int32)

    # Spin-up hopping
    for o, t, a in _compute_hopping_term(up_states, site1, site2, hop):
        origins = o * num_dn + all_dn
        targets = t * num_dn + all_dn
        for i, j in zip(origins, targets):
            yield i, j, a

    # Spin-down hopping
    for o, t, a in _compute_hopping_term(dn_states, site1, site2, hop):
        origins = all_up + o
        targets = all_up + t
        for i, j in zip(origins, targets):
            yield i, j, a


# =========================================================================
# Linear Operators
# =========================================================================


class HamiltonOperator(LinearOperator):
    """Hamiltonian as LinearOperator."""

    def __init__(self, size, data, indices, dtype=None):
        data = np.asanyarray(data)
        indices = np.asanyarray(indices)
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

    def toarray(self):
        arg1 = self.data, self.indices.T
        csr = csr_matrix(arg1, shape=self.shape, dtype=self.dtype)
        return csr.toarray()

    def _adjoint(self) -> "HamiltonOperator":
        """Hamiltonian is hermitian."""
        return self

    def _trace(self) -> float:
        """More efficient trace."""
        # Check elements where the row equals the column
        indices = np.where(self.indices[:, 0] == self.indices[:, 1])[0]
        # Return sum of diagonal elements
        return float(np.sum(self.data[indices]))

    def hash(self, algorithm="md5"):
        data = bytes(self.shape[0]) + bytes(self.indices.data) + bytes(self.data.data)
        h = hashlib.new(algorithm, data)
        return h.hexdigest()


@njit(**_jitkw)
def _apply_creation_up(matvec, x, up_states, dn_states, up_states_p1, pos):
    op = 1 << pos
    num_dn = len(dn_states)
    all_dn = np.arange(num_dn)

    # counts = np.zeros((num_dn, 1))
    # for i, dn in enumerate(dn_states):
    #     counts[i] = bit_count_between(dn, 0, pos)

    for up_idx, up in enumerate(up_states):
        if not (up & op):
            new = up ^ op
            idx_new = bisect_left(up_states_p1, new)
            origins = up_idx * num_dn + all_dn
            targets = idx_new * num_dn + all_dn
            signs = (-1) ** bit_count_between(up, 0, pos)
            matvec[targets] = signs * x[origins]


@njit(**_jitkw)
def _apply_creation_dn(matvec, x, num_sites, up_states, dn_states, dn_states_p1, pos):
    op = 1 << pos
    num_dn = len(dn_states)
    num_up = len(up_states)
    all_up = np.arange(num_up) * num_dn
    all_up_p1 = np.arange(num_up) * len(dn_states_p1)

    x_is_vec = len(x.shape) == 1
    counts = np.zeros((num_up, 1))
    for i, up in enumerate(up_states):
        counts[i] = bit_count(up, num_sites)

    for dn_idx, dn in enumerate(dn_states):
        if not (dn & op):
            new = dn ^ op
            idx_new = bisect_left(dn_states_p1, new)
            origins = all_up + dn_idx
            targets = all_up_p1 + idx_new
            signs = (-1) ** (counts + bit_count_between(dn, 0, pos))
            if x_is_vec:
                matvec[targets] = signs[:, 0] * x[origins]
            else:
                matvec[targets] = signs * x[origins]


@njit(**_jitkw)
def _apply_annihilation_up(matvec, x, up_states, dn_states_p1, up_states_p1, pos):
    op = 1 << pos
    num_dn = len(dn_states_p1)
    all_dn = np.arange(num_dn)

    for up_idx, up in enumerate(up_states_p1):
        if up & op:
            new = up ^ op
            idx_new = bisect_left(up_states, new)
            origins = up_idx * num_dn + all_dn
            targets = idx_new * num_dn + all_dn
            signs = (-1) ** bit_count_between(up, 0, pos)
            matvec[targets] = signs * x[origins]


@njit(**_jitkw)
def _apply_annihilation_dn(
    matvec, x, num_sites, up_states_p1, dn_states, dn_states_p1, pos
):
    op = 1 << pos
    num_up = len(up_states_p1)
    num_dn = len(dn_states_p1)
    all_up = np.arange(num_up) * num_dn
    all_up_p1 = np.arange(num_up) * len(dn_states)

    x_is_vec = len(x.shape) == 1
    counts = np.zeros((num_up, 1))
    for i, up in enumerate(up_states_p1):
        counts[i] = bit_count(up, num_sites)

    for dn_idx, dn in enumerate(dn_states_p1):
        if dn & op:
            new = dn ^ op
            idx_new = bisect_left(dn_states, new)
            origins = all_up + dn_idx
            targets = all_up_p1 + idx_new
            signs = (-1) ** (counts + bit_count_between(dn, 0, pos))
            if x_is_vec:
                matvec[targets] = signs[:, 0] * x[origins]
            else:
                matvec[targets] = signs * x[origins]


class CreationOperator(LinearOperator):
    """Fermionic creation operator as LinearOperator."""

    def __init__(self, sector, sector_p1, pos=0, sigma=UP):
        dim_origin = sector.size
        if sigma == UP:
            dim_target = sector_p1.num_up * sector.num_dn
        else:
            dim_target = sector_p1.num_dn * sector.num_up

        super().__init__(shape=(dim_target, dim_origin), dtype=np.float64)
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_p1 = sector_p1

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, matvec, x):
        up_states = self.sector.up_states
        dn_states = self.sector.dn_states
        up_states_p1 = self.sector_p1.up_states
        _apply_creation_up(matvec, x, up_states, dn_states, up_states_p1, self.pos)

    def _build_dn(self, matvec, x):
        nsites = self.sector.num_sites
        up_states = self.sector.up_states
        dn_states = self.sector.dn_states
        dn_states_p1 = self.sector_p1.dn_states
        _apply_creation_dn(
            matvec, x, nsites, up_states, dn_states, dn_states_p1, self.pos
        )

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

    def __init__(self, sector_m1, sector, pos=0, sigma=UP):
        dim_origin = sector.size
        if sigma == UP:
            dim_target = sector_m1.num_up * sector.num_dn
        else:
            dim_target = sector_m1.num_dn * sector.num_up

        super().__init__(shape=(dim_target, dim_origin), dtype=np.float64)
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_m1 = sector_m1

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, matvec, x):
        up_states = self.sector_m1.up_states
        up_states_p1 = self.sector.up_states
        dn_states_p1 = self.sector.dn_states
        _apply_annihilation_up(
            matvec, x, up_states, dn_states_p1, up_states_p1, self.pos
        )

    def _build_dn(self, matvec, x):
        num_sites = self.sector.num_sites
        dn_states = self.sector_m1.dn_states
        up_states_p1 = self.sector.up_states
        dn_states_p1 = self.sector.dn_states
        _apply_annihilation_dn(
            matvec, x, num_sites, up_states_p1, dn_states, dn_states_p1, self.pos
        )

    def _matvec(self, x):
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec

    def _adjoint(self):
        return CreationOperator(self.sector_m1, self.sector, self.pos, self.sigma)
