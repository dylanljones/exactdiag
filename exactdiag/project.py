# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from bisect import bisect_left
from typing import Union, Sequence
from .basis import occupations, overlap

__all__ = [
    "project_up",
    "project_dn",
    "project_elements_up",
    "project_elements_dn",
    "hubbard_interaction",
    "onsite_energy",
    "hopping",
]


def bit_count(num: int):
    try:
        return num.bit_count()  # Python 3.10 (~6 times faster)
    except AttributeError:
        return bin(num).count("1")


def project_up(
    up_idx: int, num_dn_states: int, dn_indices: Union[int, np.ndarray]
) -> np.ndarray:
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


def project_dn(
    dn_idx: int, num_dn_states: int, up_indices: Union[int, np.ndarray]
) -> np.ndarray:
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


def project_elements_up(
    up_idx: int,
    num_dn_states: int,
    dn_indices: Union[int, np.ndarray],
    value: Union[complex, float, np.ndarray],
    target: Union[int, np.ndarray] = None,
):
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
    if not value:
        return

    origins = project_up(up_idx, num_dn_states, dn_indices)
    if target is None:
        targets = origins
    else:
        targets = project_up(target, num_dn_states, dn_indices)

    for row, col in zip(origins, targets):
        yield row, col, value


def project_elements_dn(
    dn_idx: int,
    num_dn_states: int,
    up_indices: Union[int, np.ndarray],
    value: Union[complex, float, np.ndarray],
    target: Union[int, np.ndarray] = None,
):
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
    if not value:
        return

    origins = project_dn(dn_idx, num_dn_states, up_indices)
    if target is None:
        targets = origins
    else:
        targets = project_dn(target, num_dn_states, up_indices)

    for row, col in zip(origins, targets):
        yield row, col, value


# -- Interacting Hamiltonian projectors ------------------------------------------------


def onsite_energy(
    up_states: Sequence[int], dn_states: Sequence[int], eps: Sequence[float]
):
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
    >>> energies = [1.0, 2.0]
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in onsite_energy(sector.up_states, sector.dn_states, energies):
    ...     ham[i, j] += val
    >>> ham
    array([[2., 0., 0., 0.],
           [0., 3., 0., 0.],
           [0., 0., 3., 0.],
           [0., 0., 0., 4.]])

    >>> from exactdiag import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)

    """
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, up in enumerate(up_states):
        weights = occupations(up)
        energy = np.sum(eps[: weights.size] * weights)
        yield from project_elements_up(up_idx, num_dn, all_dn, energy)

    for dn_idx, dn in enumerate(dn_states):
        weights = occupations(dn)
        energy = np.sum(eps[: weights.size] * weights)
        yield from project_elements_dn(dn_idx, num_dn, all_up, energy)


def hubbard_interaction(
    up_states: Sequence[int], dn_states: Sequence[int], u: Sequence[float]
):
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
    >>> inter = [1.0, 2.0]
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in hubbard_interaction(sector.up_states, sector.dn_states, inter):
    ...     ham[i, j] += val
    >>> ham
    array([[1., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 2.]])

    >>> from exactdiag import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)

    """
    num_dn = len(dn_states)
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            weights = overlap(up, dn)
            interaction = np.sum(u[: weights.size] * weights)
            yield from project_elements_up(up_idx, num_dn, dn_idx, interaction)


def _hopping_sign(initial_state, site1, site2):
    mask = int(sum(1 << x for x in range(site1 + 1, site2)))
    jump_overs = bit_count(initial_state & mask)
    sign = (-1) ** jump_overs
    return sign


def _compute_hopping_term(states, site1, site2, hop):
    assert site1 < site2

    for i, ini in enumerate(states):
        op1 = 1 << site1  # Selects bit with index `site1`
        occ1 = ini & op1  # Value of bit of state with index `site1`
        tmp = ini ^ op1  # Annihilate/create electron at `site1`

        op2 = 1 << site2  # Selects bit with index `site2`
        occ2 = ini & op2  # Value of bit of state with index `site2`
        new = tmp ^ op2  # Create/annihilate electron at `site1`

        # ToDo: Account for hop-overs of other spin flavour
        if occ1 and not occ2:
            # Hopping from `site1` to `site2` possible
            sign = _hopping_sign(ini, site1, site2)
            j = bisect_left(states, new)
            yield i, j, sign * hop
        elif occ2 and not occ1:
            # Hopping from `site2` to `site1` possible
            sign = _hopping_sign(ini, site1, site2)
            j = bisect_left(states, new)
            yield i, j, sign * hop


def hopping(
    up_states: Sequence[int],
    dn_states: Sequence[int],
    site1: int,
    site2: int,
    hop: float,
):
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
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in hopping(sector.up_states, sector.dn_states, 0, 1, hop=1.0):
    ...     ham[i, j] += val
    >>> ham
    array([[0., 1., 1., 0.],
           [1., 0., 0., 1.],
           [1., 0., 0., 1.],
           [0., 1., 1., 0.]])

    >>> from exactdiag import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)

    """
    if site1 > site2:
        raise ValueError("The first site index must be smaller than the second one!")

    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for idx, target, amp in _compute_hopping_term(up_states, site1, site2, hop):
        yield from project_elements_up(idx, num_dn, all_dn, amp, target=target)

    for idx, target, amp in _compute_hopping_term(dn_states, site1, site2, hop):
        yield from project_elements_dn(idx, num_dn, all_up, amp, target=target)
