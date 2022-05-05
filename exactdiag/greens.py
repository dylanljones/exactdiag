# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from .basis import UP
from .operators import AnnihilationOperator, CreationOperator
from .linalg import compute_ground_state
from ._expm_multiply import expm_multiply


def _solve_sector(sector, model, cache=None):
    sector_key = (sector.n_up, sector.n_dn)
    if cache is not None and sector_key in cache:
        eigvals, eigvecs = cache[sector_key]
    else:
        sham = model.shamiltonian(sector)
        eigvals, eigvecs = np.linalg.eigh(sham.toarray())
        if cache is not None:
            cache[sector_key] = (eigvals, eigvecs)
    return eigvals, eigvecs


def _accumulate_gf(gf, z, cdag, evals, evecs, evals_p1, evecs_p1, beta, min_energy=0.0):
    cdag_vec = cdag.matmat(evecs)
    overlap = abs(evecs_p1.T.conj() @ cdag_vec) ** 2

    if np.isfinite(beta):
        exp_eigvals = np.exp(-beta * (evals - min_energy))
        exp_eigvals_p1 = np.exp(-beta * (evals_p1 - min_energy))
    else:
        exp_eigvals = np.ones_like(evals)
        exp_eigvals_p1 = np.ones_like(evals_p1)

    for m, eig_m in enumerate(evals_p1):
        for n, eig_n in enumerate(evals):
            weights = exp_eigvals[n] + exp_eigvals_p1[m]
            gf += overlap[m, n] * weights / (z + eig_n - eig_m)


def gf_z(basis, model, z, beta, pos=0, sigma=UP):
    """Computes the retarded Green's function :math:`G^{R}_{iσ}(z)`.

    Parameters
    ----------
    basis : Basis
        The many body state basis of the model.
    model : Model
        The model instance responsible for generating the Hamiltonian matrix.
    z : (...) complex array_like
        The Green's function is evaluated at these complex frequencies `z`.
    beta : float
        The inverse temperature :math:`1/k_B T`
    pos : int, optional
        The position or site index i of the Green's function
    sigma : int, optional
        The spin σ of the Green's function.

    Returns
    -------
    gf_z : (...) complex np.ndarray
        The reatrded Green's function evaluated at the complex frequencies.
    """
    gf = np.zeros_like(z)
    cache = dict()
    for sector in basis.iter_sectors():
        sector_p1 = basis.upper_sector(sector.n_up, sector.n_dn, sigma)
        if sector_p1 is not None:
            # Solve current and upper sector
            eig = _solve_sector(sector, model, cache)
            eig_p1 = _solve_sector(sector_p1, model, cache)
            # Accumulate Green's function
            cdag = CreationOperator(sector, sector_p1, pos=pos, sigma=sigma)
            _accumulate_gf(gf, z, cdag, eig[0], eig[1], eig_p1[0], eig_p1[1], beta)
        else:
            cache.clear()
    return gf


def greens_greater(basis, model, gs, start, stop, num=1000, pos=0, sigma=UP):
    """Computes the greater real time Green's function :math:`G^{>}_{iσ}(t)`.

    Parameters
    ----------
    basis : Basis
        The many body state basis of the model.
    model : Model
        The model instance responsible for generating the Hamiltonian matrix.
    gs : EigenState
        The ground state of the model.
    start : float
        The start time for computing the real time Green's function.
    stop : float
        The end time for computing the real time Green's function.
    num : int, optional
        The number of time points N used for computing the real time Green's function.
    pos : int, optional
        The position or site index i of the Green's function
    sigma : int, optional
        The spin σ of the Green's function.

    Returns
    -------
    times : (N, ) np.ndarray
        An array containing the times.
    gf_t : (N, ) np.ndarray
        The greater Green's function evaluated at the times.
    """
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = basis.get_sector(n_up, n_dn)
    times, dt = np.linspace(start, stop, num, retstep=True)
    sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
    if sector_p1 is None:
        return times, np.zeros_like(times)

    cop_dag = CreationOperator(sector, sector_p1, pos=pos, sigma=sigma)
    top_ket = cop_dag.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()  # <gs|T

    hamop = -1j * model.hamilton_operator(sector_p1)
    top_e0 = np.exp(+1j * gs.energy * dt)
    overlaps = expm_multiply(hamop, top_ket, start=start, stop=stop, num=num) @ bra_top

    factor = -1j * np.exp(+1j * gs.energy * times[0])
    overlaps[0] *= factor
    for n in range(1, num):
        factor *= top_e0
        overlaps[n] *= factor
    return times, overlaps


def greens_lesser(basis, model, gs, start, stop, num=1000, pos=0, sigma=UP):
    """Computes the lesser real time Green's function :math:`G^{<}_{iσ}(t)`.

    Parameters
    ----------
    basis : Basis
        The many body state basis of the model.
    model : Model
        The model instance responsible for generating the Hamiltonian matrix.
    gs : EigenState
        The ground state of the model.
    start : float
        The start time for computing the real time Green's function.
    stop : float
        The end time for computing the real time Green's function.
    num : int, optional
        The number of time points N used for computing the real time Green's function.
    pos : int, optional
        The position or site index i of the Green's function
    sigma : int, optional
        The spin σ of the Green's function.

    Returns
    -------
    times : (N, ) np.ndarray
        An array containing the times.
    gf_t : (N, ) np.ndarray
        The lesser Green's function evaluated at the times.
    """
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = basis.get_sector(n_up, n_dn)

    times, dt = np.linspace(start, stop, num, retstep=True)
    sector_m1 = basis.lower_sector(n_up, n_dn, sigma)
    if sector_m1 is None:
        return times, np.zeros_like(times)

    cop = AnnihilationOperator(sector, sector_m1, pos=pos, sigma=sigma)
    top_ket = cop.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()  # <gs|T

    hamop = +1j * model.hamilton_operator(sector_m1)
    top_e0 = np.exp(-1j * gs.energy * dt)
    overlaps = expm_multiply(hamop, top_ket, start=start, stop=stop, num=num) @ bra_top

    factor = +1j * np.exp(-1j * gs.energy * times[0])
    overlaps[0] *= factor
    for n in range(1, num):
        factor *= top_e0
        overlaps[n] *= factor
    return times, overlaps


def greens_function_tevo(basis, model, start, stop, num=1000, pos=0, sigma=UP):
    """Computes the real time retarded Green's function :math:`G^{R}_{iσ}(t)`.

    The retarded Green's function is the difference of the greater and lesser GF:
    .. math::
        G^{R}_{iσ}(t) = G^{>}_{iσ}(t) - G^{<}_{iσ}(t)

    Parameters
    ----------
    basis : Basis
        The many body state basis of the model.
    model : Model
        The model instance responsible for generating the Hamiltonian matrix.
    start : float
        The start time for computing the real time Green's function.
    stop : float
        The end time for computing the real time Green's function.
    num : int, optional
        The number of time points N used for computing the real time Green's function.
    pos : int, optional
        The position or site index i of the Green's function
    sigma : int, optional
        The spin σ of the Green's function.

    Returns
    -------
    times : (N, ) np.ndarray
        An array containing the times.
    gf_t : (N, ) np.ndarray
        The retarded Green's function evaluated at the times.
    """
    gs = compute_ground_state(basis, model)
    times, gf_greater = greens_greater(basis, model, gs, start, stop, num, pos, sigma)
    times, gf_lesser = greens_lesser(basis, model, gs, start, stop, num, pos, sigma)
    return times, gf_greater - gf_lesser
