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


def _build_creation_naive(sector, sector_p1, pos, sigma):
    arr = np.zeros((sector_p1.size, sector.size))
    op = 1 << pos
    for i, state_i in enumerate(sector.states):
        for j, state_j in enumerate(sector_p1.states):
            if sigma == ed.UP:
                numi_s, numj_s = state_i.up, state_j.up
                numi_stilde, numj_stilde = state_i.dn, state_j.dn
            else:
                numi_s, numj_s = state_i.dn, state_j.dn
                numi_stilde, numj_stilde = state_i.up, state_j.up

            if not (numi_s & op):
                if (numi_s ^ op) == numj_s and (numi_stilde == numj_stilde):
                    # Count bits set between 0 and pos-1 for both spin states
                    count_s = ed.operators.bit_count_between(numi_s, 0, pos)
                    count_stilde = ed.operators.bit_count_between(numi_stilde, 0, pos)
                    # Count is sum of both spin states
                    count = count_s + count_stilde
                    sign = (-1) ** count
                    arr[j, i] = sign
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
