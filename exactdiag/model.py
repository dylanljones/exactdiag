# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import json
import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator
from .operators import HamiltonOperator
from . import project


class ModelParameters(MutableMapping):
    """Parameter class for storing parameters of physical models.

    The parameters can be accessed as attributes or dict-entries.
    This class is usually used as a base-class for model-classes.
    """

    def __init__(self, **params):
        """Initializes the ModelParameters-instance.

        Parameters
        ----------
        **params
            Initial keyword parameters.
        """
        MutableMapping.__init__(self)
        self.__params__ = OrderedDict(params)

    @property
    def params(self) -> Dict[str, Any]:
        """dict: Returns a dictionary of all parameters."""
        return self.__params__

    def set_param(self, key: str, value: Any) -> None:
        """Sets a parameter

        Parameters
        ----------
        key : str
            The name of the parameter.
        value : Any
            The value of the parameter.
        """
        self.__params__[key] = value

    def delete_param(self, key: str) -> None:
        """Deletes a parameter with the given name

        Parameters
        ----------
        key : str
            The name of the parameter to delete.
        """
        del self.__params__[key]

    def rename_param(self, key: str, new_key: str) -> None:
        """Renames an existing parameter.

        Parameters
        ----------
        key : str
            The current name of the parameter.
        new_key : str
            The new name of the parameter.
        """
        self.__params__[new_key] = self.__params__[key]
        del self.__params__[key]

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self.__params__)

    def __getitem__(self, key: str) -> Any:
        """Make parameters accessable as dictionary items."""
        return self.__params__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Make parameters accessable as dictionary items."""
        self.__params__[key] = value

    def __delitem__(self, key: str) -> None:
        del self.__params__[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__params__)

    def __getattr__(self, key: str) -> Any:
        """Make parameters accessable as attributes."""
        key = str(key)
        if not key.startswith("__") and key in self.__params__.keys():
            return self.__params__[key]
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Make parameters accessable as attributes."""
        key = str(key)
        if not key.startswith("__") and key in self.__params__.keys():
            self.__params__[key] = value
        else:
            super().__setattr__(key, value)

    def __dict__(self):
        """Returns the parameters as dictionary."""
        return dict(self.__params__)

    def tostring(self, decimals: int = None, delim: str = "; ") -> str:
        """Formats the parameters as string.

        Parameters
        ----------
        decimals : int, optional
            The number of decimal places used for formatting numeric values.
        delim : str
            The delimiter used to connect the parameter strings.

        Returns
        -------
        s : str
            The formatted string.
        """
        strings = list()
        for k, v in self.__params__.items():
            if decimals is not None and isinstance(v, (int, float)):
                v = f"{v:.{decimals}f}"
            strings.append(f"{k}={v}")
        return delim.join(strings)

    def json(self) -> str:
        """Formats the parameters as JSON string.

        Returns
        -------
        s : str
            The formatted JSON string.
        """
        return json.dumps(self.__params__)

    def pformat(self) -> str:
        """Formats the parameters for printing.

        Override this function if other formatting is preferred, for example using
        greek ASCII charakters.
        """
        return ", ".join([f"{k}={v}" for k, v in self.__params__.items()])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.__dict__())})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.pformat()})"


class Model(ModelParameters, ABC):
    def __init__(self, **params):
        ModelParameters.__init__(self, **params)

    @abstractmethod
    def hamiltonian_data(self, sector):
        pass

    def hamiltonian(self, sector, dtype=None):
        size = sector.size
        ham = np.zeros((size, size), dtype=dtype)
        for i, j, val in self.hamiltonian_data(sector):
            ham[i, j] += val
        return ham

    def shamiltonian(self, sector, dtype=None):
        size = sector.size
        rows, cols, data = list(), list(), list()
        for i, j, val in self.hamiltonian_data(sector):
            rows.append(i)
            cols.append(j)
            data.append(val)
        return csr_matrix((data, (rows, cols)), shape=(size, size), dtype=dtype)

    def hamilton_operator(self, sector, dtype=None):
        size = sector.size
        rows, cols, data = list(), list(), list()
        for i, j, val in self.hamiltonian_data(sector):
            rows.append(i)
            cols.append(j)
            data.append(val)
        return HamiltonOperator(size, data, (rows, cols), dtype=dtype)


class TightBindingModel(Model):
    def __init__(self):
        pass


class HubbardModel(Model):
    def __init__(self, latt, u=0.0, eps=None, hop=1.0):
        if eps is None:
            eps = -u / 2
        super().__init__(u=u, eps=eps, hop=hop)
        self.latt = latt

    @property
    def num_sites(self):
        return self.latt.num_sites

    def hamiltonian_data(self, sector):
        num_sites = sector.num_sites
        up_states = sector.up_states
        dn_states = sector.dn_states
        energy = np.full(num_sites, fill_value=self.eps)
        hub_inter = np.full(num_sites, fill_value=self.u)

        yield from project.onsite_energy(up_states, dn_states, energy)
        yield from project.hubbard_interaction(up_states, dn_states, hub_inter)
        for i in range(self.latt.num_sites):
            for j in self.latt.nearest_neighbors(i, unique=True):
                yield from project.hopping(up_states, dn_states, i, j, self.hop)
