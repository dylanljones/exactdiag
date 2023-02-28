# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import lattpy as lp
from pytest import mark
from numpy.testing import assert_array_equal
from exactdiag.matrix import is_hermitian
from exactdiag.models import HubbardModel
import exactdiag as ed
from exactdiag import operators


def test_hubbard_sector_1_1():
    neighbors = [[0, 1]]
    model = HubbardModel(2, neighbors, inter=2.0, eps=1.0, hop=1.0)

    expected = [
        [4.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0, 4.0],
    ]
    ham = model.hamiltonian(1, 1)
    assert_array_equal(ham, expected)


@mark.parametrize("num_sites", [1, 2, 3, 4, 5, 6])
@mark.parametrize("u", [0.0, 1.0, 2.0])
def test_hamiltonian_hermitian_1d(num_sites, u):
    latt = lp.finite_hypercubic(num_sites)
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(num_sites, neighbors, inter=u, mu=u / 2, hop=1.0)
    for n_up, n_dn in model.basis.iter_fillings():
        ham = model.hamiltonian(n_up, n_dn)
        assert is_hermitian(ham)


@mark.parametrize("num_sites", [1, 2, 3, 4, 5, 6])
@mark.parametrize("u", [0.0, 1.0, 2.0])
def test_hamiltonian_hermitian_1d_periodic(num_sites, u):
    latt = lp.finite_hypercubic(num_sites, periodic=True)
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(num_sites, neighbors, inter=u, mu=u / 2, hop=1.0)
    for n_up, n_dn in model.basis.iter_fillings():
        ham = model.hamiltonian(n_up, n_dn)
        assert is_hermitian(ham)


@mark.parametrize("u", [0.0, 1.0, 2.0])
def test_hamiltonian_hermitian_2d(u):
    size = 2
    latt = lp.finite_hypercubic((size, size))
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(latt.num_sites, neighbors, inter=u, mu=u / 2, hop=1.0)
    for n_up, n_dn in model.basis.iter_fillings():
        ham = model.hamiltonian(n_up, n_dn)
        assert is_hermitian(ham)


def generate_operators(basis):
    cdg_up, cdg_dn = list(), list()
    c_up, c_dn = list(), list()
    sector = basis.get_sector()
    for i in range(basis.num_sites):
        _cdg_up = operators.CreationOperator(sector, sector, i, ed.UP).toarray()
        _cdg_dn = operators.CreationOperator(sector, sector, i, ed.DN).toarray()
        cdg_up.append(_cdg_up)
        cdg_dn.append(_cdg_dn)
        c_up.append(_cdg_up.T.conj())
        c_dn.append(_cdg_dn.T.conj())
    return cdg_up, cdg_dn, c_up, c_dn


def build_hamiltonian_ops(model):
    cdg_up, cdg_dn, c_up, c_dn = generate_operators(model.basis)

    ham = 0
    for i in range(model.num_sites):
        ham += (model.eps - model.mu) * (cdg_up[i] @ c_up[i] + cdg_dn[i] @ c_dn[i])
        ham += model.inter * (cdg_up[i] @ c_up[i] @ cdg_dn[i] @ c_dn[i])

    for i, j in model.neighbors:
        ham += model.hop * (cdg_up[i] @ c_up[j] + cdg_up[j] @ c_up[i])
        ham += model.hop * (cdg_dn[i] @ c_dn[j] + cdg_dn[j] @ c_dn[i])

    return ham


@mark.parametrize("shape", [3, 4, 5, (2, 2)])
@mark.parametrize("periodic", [False, True])
def test_hubbard_hamiltonian(shape, periodic):
    latt = lp.finite_hypercubic(shape, periodic=periodic)
    model = ed.models.HubbardModel(latt, inter=2.0, eps=1.0).hf()
    expected = build_hamiltonian_ops(model)
    ham = model.hamiltonian()
    assert_array_equal(expected, ham)
