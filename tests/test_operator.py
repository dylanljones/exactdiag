# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from pytest import mark
from hypothesis import given, strategies as st
from numpy.testing import assert_array_equal
import exactdiag as ed
from exactdiag import operators


@given(st.integers(0, 5))
def test_project_up(up_idx):
    sec = ed.Basis(5).get_sector()
    indices = [i for i, state in enumerate(sec.states) if state.up == up_idx]
    result = operators.project_up(up_idx, sec.num_dn, np.arange(sec.num_dn))
    assert_array_equal(indices, result)


@given(st.integers(0, 5))
def test_project_dn(dn_idx):
    sec = ed.Basis(5).get_sector()
    indices = [i for i, state in enumerate(sec.states) if state.dn == dn_idx]
    result = operators.project_dn(dn_idx, sec.num_dn, np.arange(sec.num_up))
    assert_array_equal(indices, result)


@mark.parametrize("num", list(range(2**8 - 1)))
def test_count_bits(num):
    num = int(num)
    expected = bin(num).count("1")
    assert operators.bit_count(num, 8) == expected


@mark.parametrize("num", list(range(2**8 - 1)))
def test_count_bits_between(num):
    num = int(num)
    for start in range(6):
        for stop in range(start, 6):
            mask = 0
            for i in range(start, stop):
                mask += 1 << i
            expected = bin(num & mask).count("1")
            assert operators.bit_count_between(num, start, stop) == expected


def op_sign(up, dn, i, sigma, num_sites):
    if sigma == ed.UP:
        count = ed.operators.bit_count_between(up, 0, i)
    else:
        count = ed.operators.bit_count_between(up, 0, num_sites)
        count += ed.operators.bit_count_between(dn, 0, i)
    return (-1) ** count


def _build_creation_naive(sector, sector_p1, pos, sigma):
    arr = np.zeros((sector_p1.size, sector.size))
    op = 1 << pos
    for i, state_i in enumerate(sector.states):
        for j, state_j in enumerate(sector_p1.states):
            up_i, up_j = state_i.up, state_j.up
            dn_i, dn_j = state_i.dn, state_j.dn
            if sigma == ed.UP and not (up_i & op):
                if (up_i ^ op) == up_j and (dn_i == dn_j):
                    arr[j, i] = op_sign(up_i, dn_i, pos, ed.UP, sector.num_sites)
            elif sigma == ed.DN and not (dn_i & op):
                if (dn_i ^ op) == dn_j and (up_i == up_j):
                    arr[j, i] = op_sign(up_i, dn_i, pos, ed.DN, sector.num_sites)
    return arr


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("sigma", [ed.UP, ed.DN])
def test_creation_operator(num_sites, sigma):
    basis = ed.Basis(num_sites)
    for n_up, n_dn in basis.iter_fillings():
        sector = basis.get_sector(n_up, n_dn)
        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            for pos in range(num_sites):
                cop_dag = operators.CreationOperator(sector, sector_p1, pos, sigma)
                expected = _build_creation_naive(sector, sector_p1, pos, sigma)
                assert_array_equal(expected, cop_dag.toarray())


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("sigma", [ed.UP, ed.DN])
def test_annihilation_operator(num_sites, sigma):
    basis = ed.Basis(num_sites)
    for n_up, n_dn in basis.iter_fillings():
        sector = basis.get_sector(n_up, n_dn)
        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            for pos in range(num_sites):
                cop = operators.AnnihilationOperator(sector, sector_p1, pos, sigma)
                expected = _build_creation_naive(sector, sector_p1, pos, sigma).T
                assert_array_equal(expected, cop.toarray())


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("sigma", [ed.UP, ed.DN])
def test_creation_annihilation_adjoint(num_sites, sigma):
    basis = ed.Basis(num_sites)
    for n_up, n_dn in basis.iter_fillings():
        sector = basis.get_sector(n_up, n_dn)
        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            for pos in range(num_sites):
                cop_dag = operators.CreationOperator(sector, sector_p1, pos, sigma)
                cop = operators.AnnihilationOperator(sector, sector_p1, pos, sigma)
                assert_array_equal(cop.toarray(), cop_dag.toarray().T.conj())
