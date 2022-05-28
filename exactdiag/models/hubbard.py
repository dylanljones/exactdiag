# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from .abc import AbstractManyBodyModel
from ..operators import (
    project_onsite_energy,
    project_hopping,
    project_hubbard_inter,
)


class HubbardModel(AbstractManyBodyModel):
    """Model class for the Hubbard model.

    The Hamiltonian of the Hubbard model is defined as
    .. math::
        H = U Σ_i n_{i↑}n_{i↓} + Σ_{iσ} ε_i c^†_{iσ}c_{iσ} + t Σ_{i,j,σ} c^†_{iσ}c_{jσ}

    Attributes
    ----------
    inter : float or Sequence, optional
        The onsite interaction energy of the model. The default value is ``0``.
    eps : float or Sequence, optional
        The onsite energy of the model. The default value is ``0``.
    eps_bath : float or Sequence, optional
        The onsite energy of the model. The default value is ``0``.
    hop : float or Sequence, optional
        The hopping parameter of the model. The default value is ``1``.
    mu : float, optional
        The chemical potential. The default is ``0``.
    """

    def __init__(self, *args, inter=None, eps=None, hop=None, mu=None):
        """Initializes the ``HubbardModel``."""
        if len(args) == 1:
            # Argument is a lattice instance
            latt = args[0]
            num_sites = latt.num_sites
            neighbors = latt.neighbor_pairs(True)[0]
        else:
            # Aurgument is the number of sites and the neighbor data
            num_sites, neighbors = args

        inter = inter or 0.0
        eps = eps or 0.0
        hop = hop or 1.0
        mu = mu or 0.0
        super().__init__(num_sites, inter=inter, eps=eps, hop=hop, mu=mu)
        self.neighbors = neighbors

    @classmethod
    def chain(cls, num_sites, inter=None, eps=None, hop=None, mu=None):
        neighbors = [[i, i + 1] for i in range(num_sites - 1)]
        return cls(num_sites, neighbors, inter=inter, eps=eps, hop=hop, mu=mu)

    def half_filling(self):
        return self.hf()

    def hf(self):
        self.set_param("mu", self.inter / 2)
        return self

    def pformat(self):
        return f"U={self.inter}, ε={self.eps}, t={self.hop}, μ={self.mu}"

    def tostring(self, decimals: int = None, delim: str = "; ") -> str:
        s = super().tostring(decimals, delim)
        return str(self.num_sites) + delim + s

    def _hamiltonian_data(self, up_states, dn_states):
        inter = self.inter
        eps = self.eps - self.mu
        hop = self.hop
        num_sites = self.num_sites
        neighbors = self.neighbors
        energy = np.full(num_sites, eps)
        interaction = np.full(num_sites, inter)

        yield from project_onsite_energy(up_states, dn_states, energy)
        yield from project_hubbard_inter(up_states, dn_states, interaction)
        for i, j in neighbors:
            yield from project_hopping(up_states, dn_states, num_sites, i, j, hop)
