# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
import lattpy as lp
import gftool as gt
from pytest import mark
from numpy.testing import assert_allclose
import exactdiag as ed
from exactdiag.greens import gf0_pole, gf_lehmann


def tight_binding_hamiltonian(latt, eps=0.0, hop=1.0):
    dmap = latt.dmap()
    data = np.zeros(dmap.size)
    data[dmap.onsite()] = eps
    data[dmap.hopping()] = hop
    return dmap.build_csr(data).toarray()


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("sigma", [ed.UP, ed.DN])
def test_gf_lehmann_hubbard_non_interacting(num_sites, sigma):
    """Test the non-interacting Hubbard Green's function at half filling.

    The many-body Green's function should be equivalent to the tight-binding
    Green's function with the same on-site and hopping energy.
    """
    beta = 100  # Low temp to match non-interacting GF
    z = np.linspace(-6, +6, 1001) + 0.05j

    latt = lp.finite_hypercubic(num_sites)
    model = ed.models.HubbardModel(latt, inter=0.0, eps=0.0, hop=1.0, beta=beta)

    ham0 = tight_binding_hamiltonian(latt, eps=model.eps, hop=model.hop)
    gf0_z_diag = gf0_pole(ham0, z=z)
    for pos in range(num_sites):
        gf0_z = gf0_z_diag[:, pos]
        gf_z = gf_lehmann(model, z, i=pos, j=pos, sigma=sigma, occ=False)[0]
        assert_allclose(gf_z, gf0_z, rtol=1e-4, atol=1e-3)


@mark.parametrize("num_sites", [2, 3])
@mark.parametrize("sigma", [ed.UP, ed.DN])
def test_gf_tevo_hubbard_non_interacting(num_sites, sigma):
    """Test the non-interacting Hubbard Green's function at half filling.

    The many-body Green's function should be equivalent to the tight-binding
    Green's function with the same on-site and hopping energy.
    """
    z = np.linspace(-5, +5, 1001) + 1e-1j
    start, stop, dt = 0.0, 200.0, 0.05
    num = int(stop // dt)

    latt = lp.finite_hypercubic(num_sites)
    model = ed.models.HubbardModel(latt, inter=0.0, eps=0.0, hop=1.0)

    ham0 = tight_binding_hamiltonian(latt, eps=model.eps, hop=model.hop)
    gf0_z_diag = gf0_pole(ham0, z=z, mode="diag")
    for pos in range(num_sites):
        gf0_z = gf0_z_diag[:, pos]

        t, gf_tevo = ed.gf_tevo(model, start, stop, num, pos, sigma=sigma)
        gf_z = gt.fourier.tt2z(t, gf_tevo, z)
        assert_allclose(gf_z, gf0_z, rtol=1e-4, atol=1e-2)


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("u", [1.0, 2.0, 3.0, 4.0])
@mark.parametrize("sigma", [ed.UP, ed.DN])
def test_gf_lehmann_hubbard_atomic_limit(num_sites, u, sigma):
    """Test the Hubbard Green's function at half filling with no hopping.

    The many-body Green's function should only have two peaks at -u/2 and +u/2.
    """
    beta = 100.0
    z = np.linspace(-4, +4, 1001) + 0.05j

    latt = lp.finite_hypercubic(num_sites)
    model = ed.models.HubbardModel(latt, inter=u, eps=0.0, hop=0.0, beta=beta).hf()
    for pos in range(num_sites):
        gf_z = gf_lehmann(model, z, i=pos, j=pos, sigma=sigma, occ=False)[0]

        c = len(z) // 2
        gf_neg = -gf_z.imag[:c]
        ww_neg = z.real[:c]
        gf_pos = -gf_z.imag[c:]
        ww_pos = z.real[c:]
        # z < 0
        peak0 = np.argmax(gf_neg)
        energy0 = ww_neg[peak0]
        # z > 0
        peak1 = np.argmax(gf_pos)
        energy1 = ww_pos[peak1]

        assert abs(energy0 - (-u / 2)) < 0.1
        assert abs(energy1 - (+u / 2)) < 0.1


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("u", [1.0, 2.0, 3.0, 4.0])
def test_gf_lehmann_hubbard_occupation(num_sites, u):
    """Test the occupation of the Hubabrd model.

    At half filling this should always be 0.5
    """
    beta = 100.0
    z = np.linspace(-4, +4, 1001) + 0.05j

    latt = lp.finite_hypercubic(num_sites)
    model = ed.models.HubbardModel(latt, inter=u, eps=0.0, hop=1.0, beta=beta).hf()
    for pos in range(num_sites):
        occ = gf_lehmann(model, z, i=pos, j=pos, occ=True)[1]
        assert abs(occ - 0.5) < 1e-3
