# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import os
import pickle
import shutil
import logging
import numpy as np
from numba import njit, prange
from typing import MutableMapping
from .basis import Sector, UP
from .models import AbstractManyBodyModel
from .operators import AnnihilationOperator, CreationOperator
from .linalg import compute_ground_state
from ._expm_multiply import expm_multiply

# from scipy.sparse.linalg import expm_multiply
logger = logging.getLogger(__name__)

_jitkw = dict(fastmath=True, nogil=True, parallel=True, cache=True)


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


def solve_sector(
    model: AbstractManyBodyModel, sector: Sector, cache: MutableMapping = None
):
    """Solves the eigenvalue problem for a sector of a many-body model.

    Parameters
    ----------
    model : AbstractManyBodyModel
        The N-site many-body model. It is responsible for generating the Hamilton
        operator.
    sector : Sector
        The basis sector to solve.
    cache : MutableMapping
        A cache used for storing the eigenvalues and eigenvectors. The hash of the
        Hamilton operator is used as keys.

    Returns
    -------
    eigvals : (N, ) np.ndarray
        The eigenvalues of the sector.
    eigenvecs : (N, N) np.ndarray
        The eigenvectors of the sector.
    """
    hamop = model.hamilton_operator(sector=sector)
    if cache is None:
        return np.linalg.eigh(hamop.toarray())

    key = hamop.hash()
    try:
        eigvals, eigvecs = cache[key]
        logger.debug("Loading eig  %d, %d (%s)", sector.n_up, sector.n_dn, sector.size)
    except KeyError:
        logger.debug("Solving eig  %d, %d (%s)", sector.n_up, sector.n_dn, sector.size)
        eigvals, eigvecs = np.linalg.eigh(hamop.toarray())
        cache[key] = (eigvals, eigvecs)

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


@njit("void(c16[:], c16[:], f8[:], f8[:], f8[:,::1], f8[:,::1], f8, f8)", **_jitkw)
def _acc_gf_diag(gf, z, evals, evals_p1, evecs_p1, cdag_evec, beta, emin):
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


@njit(
    "void(c16[:],c16[:],f8[:,::1],f8[:,::1],f8[:],f8[:,::1],f8[:],f8[:,::1],f8,f8)",
    **_jitkw,
)
def _acc_gf(gf, z, c_evec_m, cdag_evec_n, evals_n, evecs_n, evals_m, evecs_m, beta, e0):
    overlap1 = evecs_n.T.conj() @ c_evec_m  # <n|c_i|m>
    overlap2 = evecs_m.T.conj() @ cdag_evec_n  # <m|c_j^†|n>

    if np.isfinite(beta):
        exp_n = np.exp(-beta * (evals_n - e0))
        exp_m = np.exp(-beta * (evals_m - e0))
    else:
        exp_n = np.ones_like(evals_n)
        exp_m = np.ones_like(evals_m)

    num_n, num_m = len(evals_n), len(evals_m)
    for m in prange(num_m):
        z_m = z - evals_m[m]
        for n in range(num_n):
            overlap = overlap1[n, m] * overlap2[m, n]
            weights = exp_n[n] + exp_m[m]
            gf += overlap * weights / (z_m + evals_n[n])


def accumulate_gf_diag(gf, z, cdag, evals, evecs, evals_p1, evecs_p1, beta, emin=0.0):
    cdag_evec = cdag.matmat(evecs)
    _acc_gf_diag(gf, z, evals, evals_p1, evecs_p1, cdag_evec, beta, emin)


def accumulate_gf(gf, z, cop, cdag, evals, evecs, evals_p1, evecs_p1, beta, emin=0.0):
    c_ev_m = cop.matmat(evecs_p1)
    cd_ev_n = cdag.matmat(evecs)
    _acc_gf(gf, z, c_ev_m, cd_ev_n, evals, evecs, evals_p1, evecs_p1, beta, emin)


class GreensFunctionMeasurement:
    def __init__(self, z, beta, i=0, j=0, sigma=UP, dtype=None, measure_occ=True):
        self.z = z
        self.beta = beta
        self.i = i
        self.j = j
        self.sigma = sigma
        self._measure_occ = measure_occ

        self._part = 0
        self._gs_energy = np.infty
        self._gf = np.zeros_like(z, dtype=dtype)
        self._occ = np.nan if i != j else 0.0
        self._occ_double = np.nan if i != j else 0.0

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

        gf = self._gf
        z = self.z
        beta = self.beta
        e0 = self._gs_energy
        cdg = CreationOperator(sector, sector_p1, pos=self.j, sigma=self.sigma)
        if self.i == self.j:
            accumulate_gf_diag(gf, z, cdg, evals, evecs, evals_p1, evecs_p1, beta, e0)
        else:
            c = AnnihilationOperator(sector, sector_p1, pos=self.i, sigma=self.sigma)
            accumulate_gf(gf, z, c, cdg, evals, evecs, evals_p1, evecs_p1, beta, e0)

    def _acc_occ(self, up, dn, evals, evecs, factor):
        beta = self.beta
        e0 = self._gs_energy
        self._occ *= factor
        self._occ += occupation(up, dn, evals, evecs, beta, e0, self.i, self.sigma)

    def _acc_occ_double(self, up, dn, evals, evecs, factor):
        beta = self.beta
        e0 = self._gs_energy
        self._occ_double *= factor
        self._occ_double += double_occupation(up, dn, evals, evecs, beta, e0, self.i)

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
        if self._measure_occ and self.i == self.j:
            self._acc_occ(up, dn, evals, evecs, factor)
            self._acc_occ_double(up, dn, evals, evecs, factor)

    def save(self, file):
        data = [self._part, self._gs_energy, self._gf, self._occ, self._occ_double]
        with open(file, "wb") as fh:
            pickle.dump(data, fh)

    def load(self, file):
        with open(file, "rb") as fh:
            data = pickle.load(fh)
        self._part = data[0]
        self._gs_energy = data[1]
        self._gf = data[2]
        self._occ = data[3]
        self._occ_double = data[4]


def gf_lehmann(
    model, z, i=0, j=None, sigma=UP, eig_cache=None, occ=True, cache_dir=None
):
    if j is None:
        j = i

    basis = model.basis
    eig_cache = eig_cache if eig_cache is not None else dict()

    logger.info("Accumulating Lehmann sum (i=%s, j=%s, sigma=%s)", i, j, sigma)
    logger.debug("Sites: %s (%s states)", basis.num_sites, basis.size)

    data = GreensFunctionMeasurement(z, model.beta, i, j, sigma, measure_occ=occ)
    fillings = list(basis.iter_fillings())
    num = len(fillings)
    w = len(str(num))

    if cache_dir is not None and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for it, (n_up, n_dn) in enumerate(fillings):
        sector = model.get_sector(n_up, n_dn)
        logger.info("[%s/%s] Sector %s, %s", f"{it+1:>{w}}", num, n_up, n_dn)

        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            if cache_dir is not None:
                file = os.path.join(cache_dir, f"sector_{n_up}_{n_dn}")
            else:
                file = ""

            if os.path.exists(file):
                data.load(file)
            else:
                evals, evecs = solve_sector(model, sector, cache=eig_cache)
                evals_p1, evecs_p1 = solve_sector(model, sector_p1, cache=eig_cache)
                data.accumulate(sector, sector_p1, evals, evecs, evals_p1, evecs_p1)
                if file:
                    data.save(file)
        else:
            logger.debug("No upper sector, skipping")

    logger.info("gs-energy:   %+.4f", data.gs_energy)
    if occ:
        logger.info("occupation:  %.4f", data.occ)
        logger.info("double-occ:  %.4f", data.occ_double)
    return data.gf, data.occ, data.occ_double


def _compute_gf(model, z, i, j, sigma, eig_cache=None, directory=None):
    if directory is None:
        return gf_lehmann(model, z, i, j, sigma, eig_cache)[0]

    model_hash = model.hash()
    # Cache directory (stores sector GFs)
    cache_dir = os.path.join(directory, model_hash, f"sector_{i}_{j}_{sigma}")
    # File path for storing results of G_{ijs}
    file = os.path.join(directory, model_hash, f"gdat_{i}_{j}_{sigma}")

    if os.path.exists(file):
        # Load stored data
        gf = np.loadtxt(file, dtype=np.complex128)
    else:
        # Compute data and store result
        gf = gf_lehmann(model, z, i, j, sigma, eig_cache, cache_dir=cache_dir)[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(file, gf)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    return gf


def compute_gf_diag(model, z, sigma=UP, eig_cache=None, directory=None):
    eig_cache = dict() if eig_cache is None else eig_cache
    data = np.zeros((len(z), model.num_sites), dtype=np.complex128)
    for i in range(model.num_sites):
        data[:, i] = _compute_gf(model, z, i, i, sigma, eig_cache, directory)
    return data


def compute_gf_full(model, z, sigma=UP, eig_cache=None, directory=None):
    eig_cache = dict() if eig_cache is None else eig_cache
    num = model.num_sites
    data = np.zeros((len(z), num, num), dtype=np.complex128)
    for i in range(num):
        for j in range(i, num):
            gf = _compute_gf(model, z, i, j, sigma, eig_cache, directory)
            data[:, i, j] = gf
            if i < j:
                data[:, j, i] = gf
    return data


def gf_greater(model, gs, start, stop, num=1000, pos=0, sigma=UP):
    """Computes the greater real time Green's function :math:`G^{>}_{iσ}(t)`.

    Parameters
    ----------
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
    basis = model.basis
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = basis.get_sector(n_up, n_dn)
    sector_p1 = basis.upper_sector(n_up, n_dn, sigma)

    times, dt = np.linspace(start, stop, num, retstep=True)
    if sector_p1 is None:
        return times, np.zeros_like(times)

    cop_dag = CreationOperator(sector, sector_p1, pos=pos, sigma=sigma)
    top_ket = cop_dag.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()  # <gs|T

    hamop = -1j * model.hamilton_operator(sector=sector_p1)
    top_e0 = np.exp(+1j * gs.energy * dt)
    overlaps = (
        expm_multiply(
            hamop, top_ket, traceA=hamop.trace(), start=start, stop=stop, num=num
        )
        @ bra_top
    )

    factor = -1j * np.exp(+1j * gs.energy * times[0])
    overlaps[0] *= factor
    for n in range(1, num):
        factor *= top_e0
        overlaps[n] *= factor
    return times, overlaps


def gf_lesser(model, gs, start, stop, num=1000, pos=0, sigma=UP):
    """Computes the lesser real time Green's function :math:`G^{<}_{iσ}(t)`.

    Parameters
    ----------
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
    basis = model.basis
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = basis.get_sector(n_up, n_dn)
    sector_m1 = basis.lower_sector(n_up, n_dn, sigma)

    times, dt = np.linspace(start, stop, num, retstep=True)
    if sector_m1 is None:
        return times, np.zeros_like(times)

    cop = AnnihilationOperator(sector_m1, sector, pos=pos, sigma=sigma)
    top_ket = cop.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()  # <gs|T

    hamop = +1j * model.hamilton_operator(sector=sector_m1)
    top_e0 = np.exp(-1j * gs.energy * dt)
    overlaps = (
        expm_multiply(
            hamop, top_ket, start=start, stop=stop, num=num, traceA=hamop.trace()
        )
        @ bra_top
    )

    factor = +1j * np.exp(-1j * gs.energy * times[0])
    overlaps[0] *= factor
    for n in range(1, num):
        factor *= top_e0
        overlaps[n] *= factor
    return times, overlaps


def gf_tevo(model, start, stop, num=1000, pos=0, sigma=UP):
    """Computes the real time retarded Green's function :math:`G^{R}_{iσ}(t)`.

    The retarded Green's function is the difference of the greater and lesser GF:
    .. math::
        G^{R}_{iσ}(t) = G^{>}_{iσ}(t) - G^{<}_{iσ}(t)

    Parameters
    ----------
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
    gs = compute_ground_state(model.basis, model)
    times, gf_g = gf_greater(model, gs, start, stop, num, pos, sigma)
    times, gf_l = gf_lesser(model, gs, start, stop, num, pos, sigma)
    return times, gf_g - gf_l
