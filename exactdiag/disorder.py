# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2023, Dylan Jones

import hashlib
from abc import ABC, abstractmethod
import numpy as np


class DisorderGenerator(ABC):
    def __init__(self, size, seed=None):
        if seed is None:
            seed = np.random.randint(0, 65536)
        self.size = size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def disorder(self, *args, **kwargs):
        pass

    def generate(self, num, *args, **kwargs):
        for _ in range(num):
            yield self.disorder(*args, **kwargs)

    def realizations(self, num, *args, **kwargs):
        return np.array(list(self.generate(num, *args, **kwargs)))

    def __call__(self, *args, **kwargs):
        return self.disorder(*args, **kwargs)

    def __iter__(self):
        while True:
            yield self.disorder()

    def hash(self):
        s = f"{self.__class__.__name__}-{self.size}-{self.seed}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def __hash__(self):
        return self.hash()


class DiagonalDisorder(DisorderGenerator):
    """Generator for uniform diagonal disorder."""

    def __init__(self, size, strength=1.0, loc=0.0, seed=None, distribution="uniform"):
        super().__init__(size, seed)
        self.loc = loc
        self.strength = strength
        self.distribution = distribution

    def disorder(self, *args, **kwargs):
        if self.distribution == "uniform":
            de = self.strength / 2
            return self.rng.uniform(self.loc - de, self.loc + de, size=self.size)
        elif self.distribution == "normal":
            return self.rng.normal(self.loc, self.strength, size=self.size)
        else:
            return ValueError(f"Distribution '{self.distribution}' not supported.")


class NormalDisorder(DisorderGenerator):
    """Generator for disorder with a normal probability distribution."""

    def __init__(self, size, loc=0.0, scale=1.0, seed=None):
        super().__init__(size, seed)
        self.loc = loc
        self.scale = scale

    def set_loc(self, loc):
        self.loc = loc

    def set_scale(self, scale):
        self.scale = scale

    def disorder(self):
        return self.rng.normal(self.loc, self.scale, size=self.size)


class SubstDiagonalDisorder(DiagonalDisorder):
    """Generator for substitutional dirsorder."""

    def __init__(self, size, energies, concentrations, seed=None):
        super().__init__(size, seed)
        self.energies = energies
        self.concentrations = concentrations

    def disorder(self):
        return self.rng.choice(self.energies, size=self.size, p=self.concentrations)


class BinaryDiagonalDisorder(SubstDiagonalDisorder):
    """Generator for binary substitutional dirsorder."""

    def __init__(self, size, eps_a, eps_b, c_b, seed=None):
        super().__init__(size, [eps_a, eps_b], [1 - c_b, c_b], seed)
