# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from exactdiag import UP, CreationOperator


def solve_sector(sector, sham_func, *args, cache=None, **kwargs):
    sector_key = (sector.n_up, sector.n_dn)
    if cache is not None and sector_key in cache:
        eigvals, eigvecs = cache[sector_key]
    else:
        sham = sham_func(sector, *args, **kwargs)
        eigvals, eigvecs = np.linalg.eigh(sham.toarray())
        if cache is not None:
            cache[sector_key] = (eigvals, eigvecs)
    return eigvals, eigvecs


def accumulate_gf(
    gf, z, cdag, eigvals, eigvecs, eigvals_p1, eigvecs_p1, beta, min_energy=0.0
):
    cdag_vec = cdag.matmat(eigvecs)
    overlap = abs(eigvecs_p1.T.conj() @ cdag_vec) ** 2

    if np.isfinite(beta):
        exp_eigvals = np.exp(-beta * (eigvals - min_energy))
        exp_eigvals_p1 = np.exp(-beta * (eigvals_p1 - min_energy))
    else:
        exp_eigvals = np.ones_like(eigvals)
        exp_eigvals_p1 = np.ones_like(eigvals_p1)

    for m, eig_m in enumerate(eigvals_p1):
        for n, eig_n in enumerate(eigvals):
            weights = exp_eigvals[n] + exp_eigvals_p1[m]
            gf += overlap[m, n] * weights / (z + eig_n - eig_m)


def gf_lehmann(basis, z, beta, sham_func, *args, pos=0, sigma=UP, cache=None, **kwargs):
    gf = np.zeros_like(z)
    cache = cache if cache is not None else dict()
    for sector in basis.iter_sectors():
        sector_p1 = basis.upper_sector(sector.n_up, sector.n_dn, sigma)
        if sector_p1 is not None:
            eig = solve_sector(sector, sham_func, *args, **kwargs)
            eig_p1 = solve_sector(sector_p1, sham_func, *args, **kwargs)
            cdag = CreationOperator(sector, sector_p1, pos=pos, sigma=sigma)
            accumulate_gf(gf, z, cdag, eig[0], eig[1], eig_p1[0], eig_p1[1], beta)
        else:
            cache.clear()
    return gf
