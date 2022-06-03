# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from .abc import AbstractManyBodyModel
from ..operators import project_onsite_energy, project_hopping, project_hubbard_inter


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

    def __init__(self, *args, inter=None, eps=None, hop=None, mu=None, beta=1.0):
        """Initializes the ``HubbardModel``."""
        if len(args) == 1:
            # Argument is a lattice instance
            latt = args[0]
            num_sites = latt.num_sites
            neighbors = latt.neighbor_pairs(unique=True)[0]
        else:
            # Aurgument is the number of sites and the neighbor data
            num_sites, neighbors = args

        inter = inter if inter is not None else 0.0
        eps = eps if eps is not None else 0.0
        hop = hop if hop is not None else 1.0
        mu = mu if mu is not None else 0.0
        super().__init__(num_sites, inter=inter, eps=eps, hop=hop, mu=mu, beta=beta)
        self.neighbors = neighbors

    @classmethod
    def chain(cls, num_sites, **kwargs):
        neighbors = [[i, i + 1] for i in range(num_sites - 1)]
        return cls(num_sites, neighbors, **kwargs)

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
        energy = np.full(num_sites, eps, dtype=np.float64)
        interaction = np.full(num_sites, inter, dtype=np.float64)

        yield from project_onsite_energy(up_states, dn_states, energy)
        yield from project_hubbard_inter(up_states, dn_states, interaction)
        for i, j in neighbors:
            yield from project_hopping(up_states, dn_states, i, j, float(hop))

    def _hamiltonian_data0(self):
        for i in range(self.num_sites):
            yield i, i, self.eps
        for i, j in self.neighbors:
            yield i, j, self.hop
            yield j, i, self.hop
