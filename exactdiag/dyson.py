# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2023, Dylan Jones

from numpy import linalg as la


def dyson_self_energy(gf0, gf):
    """Computes the self energy Σ via the Dyson equation

    The Dyson equation for computing the self energy Σ is given by:
    .. math::
        Σ(z) = G_0(z)^{-1} - G(z)^{-1}

    Parameters
    ----------
    gf0 : complex np.ndarray
        The unperturbed Green's function .math:`G_0(z)` evaluated at complex
        frequency `z`.
    gf : complex np.ndarray
        The full Green's function .math:`G(z)` evaluated at complex frequency `z`.

    Returns
    -------
    sigma : complex np.ndarray
        The self energy `Σ(z)` at complex frequency `z`.

    Notes
    -----
    The Green's function's can be represented as full matrix (..., N, N),
    diagonal elements (..., N) or as the trace (...), where (...) is the shape of the
    frequencies `z`.
    """
    if len(gf.shape) > 2:
        # Array of matrices
        sigma = la.inv(gf0) - la.inv(gf)
    else:
        # Array traced matrices
        sigma = (1 / gf0) - (1 / gf)
    return sigma


def dyson_greens_function(gf0, sigma):
    """Computes the full green's function G via the Dyson equation.

    The Dyson equation for computing the Green's function G is given by:
    .. math::
        G(z) = [G_0(z)^{-1} - Σ(z)]^{-1}

    Parameters
    ----------
    gf0 : complex np.ndarray
        The unperturbed Green's function .math:`G_0(z)` evaluated at complex
        frequency `z`.
    sigma : complex np.ndarray
        The self energy `Σ` evaluated at complex frequency `z`.

    Returns
    -------
    gf : complex np.ndarray
        The full Green's function .math:`G(z)` at complex frequency `z`.

    Notes
    -----
    The Green's function and self energy can be represented as full matrix (..., N, N),
    diagonal elements (..., N) or as the trace (...), where (...) is the shape of the
    frequencies `z`.
    """
    if len(gf0.shape) > 2:
        gf = la.inv(la.inv(gf0) - sigma)
    else:
        gf = 1 / ((1 / gf0) - sigma)
    return gf
