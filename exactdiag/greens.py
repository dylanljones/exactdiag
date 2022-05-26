# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import logging
import numpy as np
from numba import njit, prange
from .basis import Sector, UP
from .models import AbstractManyBodyModel
from .operators import AnnihilationOperator, CreationOperator
from .linalg import compute_ground_state
from ._expm_multiply import expm_multiply

logger = logging.getLogger(__name__)

_jitkw = dict(fastmath=True, nogil=True, parallel=True)


def gf0_pole(*args, z, mode="diag") -> np.ndarray:
    """Calculate the non-interacting Green's function.

    Parameters
    ----------
    *args : tuple of np.ndarray
        Input argument. This can either be a tuple of size two, containing arrays of
        eigenvalues and eigenvectors or a single argument, interpreted as
        Hamilton-operator and used to compute the eigenvalues and eigenvectors used in
        the calculation. The eigenvectors of the Hamiltonian.
    z : (..., Nw) complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    mode : str, optional
        The output mode of the method. Can either be 'full', 'diag' or 'total'.
        The default is 'diag'. Mode 'full' computes the full Green's function matrix,
        'diag' the diagonal and 'total' computes the trace of the Green's function.

    Returns
    -------
    gf : (...., Nw, N) complex np.ndarray or (...., Nw, N, N) complex np.ndarray
        The Green's function evaluated at `z`.
    """
    if len(args) == 1:
        eigvals, eigvecs = np.linalg.eigh(args[0])
    else:
        eigvals, eigvecs = args

    z = np.atleast_1d(z)
    eigvecs_adj = np.conj(eigvecs).T

    if mode == "full":
        subscript_str = "ik,...k,kj->...ij"
    elif mode == "diag":
        subscript_str = "ij,...j,ji->...i"
    elif mode == "total":
        subscript_str = "ij,...j,ji->..."
    else:
        raise ValueError(
            f"Mode '{mode}' not supported. "
            f"Valid modes are 'full', 'diag' or 'total'"
        )
    arg = np.subtract.outer(z, eigvals)
    return np.einsum(subscript_str, eigvecs_adj, 1 / arg, eigvecs)


def solve_sector(model: AbstractManyBodyModel, sector: Sector, cache: dict = None):
    sector_key = (sector.n_up, sector.n_dn)
    if cache is not None and sector_key in cache:
        logger.debug("Loading eig  %d, %d (%s)", sector.n_up, sector.n_dn, sector.size)
        eigvals, eigvecs = cache[sector_key]
    else:
        logger.debug("Solving eig  %d, %d (%s)", sector.n_up, sector.n_dn, sector.size)
        ham = model.hamiltonian(sector=sector)
        eigvals, eigvecs = np.linalg.eigh(ham)
        if cache is not None:
            cache[sector_key] = [eigvals, eigvecs]
    return eigvals, eigvecs


@njit("f8(i8[:], i8[:], f8[:], f8[:, :], f8, f8, i8)", **_jitkw)
def occupation_up(up_states, dn_states, evals, evecs, beta, emin, pos):
    num_dn = len(dn_states)
    all_dn = np.arange(num_dn)
    occ = 0.0
    for up_idx in prange(len(up_states)):
        up = up_states[up_idx]
        if up & (1 << pos):  # state occupied
            indices = up_idx * num_dn + all_dn
            overlap = np.sum(np.abs(evecs[indices, :]) ** 2, axis=0)
            occ += np.sum(np.exp(-beta * (evals - emin)) * overlap)
    return occ


@njit("f8(i8[:], i8[:], f8[:], f8[:, :], f8, f8, i8)", **_jitkw)
def occupation_dn(up_states, dn_states, evals, evecs, beta, emin, pos):
    num_dn = len(dn_states)
    all_up = np.arange(len(up_states))
    occ = 0.0
    for dn_idx in prange(num_dn):
        dn = dn_states[dn_idx]
        if dn & (1 << pos):  # state occupied
            indices = all_up * num_dn + dn_idx
            overlap = np.sum(np.abs(evecs[indices, :]) ** 2, axis=0)
            occ += np.sum(np.exp(-beta * (evals - emin)) * overlap)
    return occ


def occupation(up_states, dn_states, evals, evecs, beta, emin=0.0, pos=0, sigma=UP):
    if sigma == UP:
        return occupation_up(up_states, dn_states, evals, evecs, beta, emin, pos)
    else:
        return occupation_dn(up_states, dn_states, evals, evecs, beta, emin, pos)


@njit("f8(i8[:], i8[:], f8[:], f8[:, :], f8, f8, i8)", **_jitkw)
def double_occupation(up_states, dn_states, evals, evecs, beta, emin, pos):
    occ = 0.0
    idx = 0
    for up_idx in prange(len(up_states)):
        for dn_idx in range(len(dn_states)):
            up = up_states[up_idx]
            dn = dn_states[dn_idx]
            if up & dn & (1 << pos):
                overlap = np.abs(evecs[idx, :]) ** 2
                occ += np.sum(np.exp(-beta * (evals - emin)) * overlap)
            idx += 1
    return occ


@njit("void(c16[:], c16[:], f8[:], f8[:], f8[:, :], f8[:, :], f8, f8)", **_jitkw)
def _accumulate_sum(gf, z, evals, evals_p1, evecs_p1, cdag_evec, beta, emin):
    overlap = np.abs(evecs_p1.T.conj() @ cdag_evec) ** 2

    if np.isfinite(beta):
        exp_evals = np.exp(-beta * (evals - emin))
        exp_evals_p1 = np.exp(-beta * (evals_p1 - emin))
    else:
        exp_evals = np.ones_like(evals)
        exp_evals_p1 = np.ones_like(evals_p1)

    num_m = len(evals_p1)
    num_n = len(evals)
    for m in prange(num_m):
        eig_m = evals_p1[m]
        z_m = z - eig_m
        for n in range(num_n):
            eig_n = evals[n]
            weights = exp_evals[n] + exp_evals_p1[m]
            gf += overlap[m, n] * weights / (z_m + eig_n)


def accumulate_gf(gf, z, cdag, evals, evecs, evals_p1, evecs_p1, beta, emin=0.0):
    cdag_evec = cdag.matmat(evecs)
    return _accumulate_sum(gf, z, evals, evals_p1, evecs_p1, cdag_evec, beta, emin)


class GreensFunctionMeasurement:
    def __init__(self, z, beta, pos=0, sigma=UP, dtype=None, measure_occ=True):
        self.z = z
        self.beta = beta
        self.pos = pos
        self.sigma = sigma
        self._measure_occ = measure_occ

        self._part = 0
        self._gs_energy = np.infty
        self._gf = np.zeros_like(z, dtype=dtype)
        self._occ = 0.0
        self._occ_double = 0.0

    @property
    def part(self):
        return self._part * np.exp(-self.beta * self._gs_energy)

    @property
    def gf(self):
        return self._gf / self._part

    @property
    def occ(self):
        return self._occ / self._part

    @property
    def occ_double(self):
        return self._occ_double / self._part

    @property
    def gs_energy(self):
        return self._gs_energy

    def _acc_part(self, eigvals, factor=1.0):
        self._part *= factor
        self._part += np.sum(np.exp(-self.beta * (eigvals - self._gs_energy)))

    def _acc_gf(self, sector, sector_p1, evals, evecs, evals_p1, evecs_p1, factor):
        if factor != 1.0:
            self._gf *= factor

        cdag = CreationOperator(sector, sector_p1, pos=self.pos, sigma=self.sigma)
        z = self.z
        beta = self.beta
        e0 = self._gs_energy
        accumulate_gf(self._gf, z, cdag, evals, evecs, evals_p1, evecs_p1, beta, e0)

    def _acc_occ(self, up, dn, evals, evecs, factor):
        beta = self.beta
        e0 = self._gs_energy
        self._occ *= factor
        self._occ += occupation(up, dn, evals, evecs, beta, e0, self.pos, self.sigma)

    def _acc_occ_double(self, up, dn, evals, evecs, factor):
        beta = self.beta
        e0 = self._gs_energy
        self._occ_double *= factor
        self._occ_double += double_occupation(up, dn, evals, evecs, beta, e0, self.pos)

    def accumulate(self, sector, sector_p1, evals, evecs, evals_p1, evecs_p1):
        min_energy = min(evals)
        factor = 1.0
        if min_energy < self._gs_energy:
            factor = np.exp(-self.beta * (self._gs_energy - min_energy))
            self._gs_energy = min_energy
            logger.debug("New ground state: E_0=%.4f", min_energy)

        logger.debug("Accumulating")
        up = np.array(sector.up_states, dtype=np.int64)
        dn = np.array(sector.dn_states, dtype=np.int64)
        self._acc_part(evals, factor)
        self._acc_gf(sector, sector_p1, evals, evecs, evals_p1, evecs_p1, factor)
        if self._measure_occ:
            self._acc_occ(up, dn, evals, evecs, factor)
            self._acc_occ_double(up, dn, evals, evecs, factor)


def gf_lehmann(model, z, beta, pos=0, sigma=UP, eig_cache=None, occ=True):
    basis = model.basis

    logger.info("Accumulating Lehmann sum (pos=%s, sigma=%s)", pos, sigma)
    logger.debug("Sites: %s (%s states)", basis.num_sites, basis.size)

    data = GreensFunctionMeasurement(z, beta, pos, sigma, measure_occ=occ)
    eig_cache = eig_cache if eig_cache is not None else dict()

    fillings = list(basis.iter_fillings())
    num = len(fillings)
    w = len(str(num))
    for i, (n_up, n_dn) in enumerate(fillings):
        sector = model.get_sector(n_up, n_dn)
        logger.info("[%s/%s] Sector %s, %s", f"{i+1:>{w}}", num, n_up, n_dn)

        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            eigvals, eigvecs = solve_sector(model, sector, cache=eig_cache)
            eigvals_p1, eigvecs_p1 = solve_sector(model, sector_p1, cache=eig_cache)
            data.accumulate(sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1)
        else:
            logger.debug("No upper sector, skipping")
            # eig_cache.clear()

    logger.info("-" * 40)
    logger.info("gs-energy:  %+.4f", data.gs_energy)
    logger.info("occupation:  %.4f", data.occ)
    logger.info("double-occ:  %.4f", data.occ_double)
    logger.info("-" * 40)
    return data


def gf_greater(basis, model, gs, start, stop, num=1000, pos=0, sigma=UP):
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


def gf_lesser(basis, model, gs, start, stop, num=1000, pos=0, sigma=UP):
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


def gf_tevo(basis, model, start, stop, num=1000, pos=0, sigma=UP):
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
    times, gf_g = gf_greater(basis, model, gs, start, stop, num, pos, sigma)
    times, gf_l = gf_lesser(basis, model, gs, start, stop, num, pos, sigma)
    return times, gf_g - gf_l
