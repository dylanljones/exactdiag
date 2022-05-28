# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import sys
import os
import h5py
import math
import random
import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import abc, OrderedDict
from .utils import CACHE_DIR

logger = logging.getLogger(__name__)

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


def formatnum(number, unit="", div=1000.0, frmt="", delim=""):
    unit_prefix = ["", "k", "M", "G", "T", "P"]
    k = int(math.floor(math.log(number, div)))
    return f"{number / div**k:{frmt}}{delim}{unit_prefix[k]}{unit}"


def formatsize(number, dec=1, delim=""):
    div = 1024.0
    unit = "iB"
    if number <= div:
        return f"{number:.0f}{delim}B"
    return formatnum(number, unit, div, frmt=f".{dec}f", delim=delim)


def getsizeof(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([getsizeof(v, seen) for v in obj.values()])
        size += sum([getsizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += getsizeof(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([getsizeof(i, seen) for i in obj])
    return size


class AbstractCache(abc.MutableMapping, ABC):

    RAND = "RAND"
    FIFO = "FIFO"
    LRU = "LRU"

    def __init__(self, maxcount=0, maxsize=0, policy="FIFO"):
        self._enabled = True
        self._policy = policy.upper()
        self._maxcount = maxcount
        self._maxsize = maxsize

    @abstractmethod
    def __sizeof__(self):
        pass

    @property
    def enabled(self):
        return self._enabled

    @property
    def count(self):
        return self.__len__()

    @property
    def size(self):
        return self.__sizeof__()

    def set_enabled(self, b):
        self._enabled = bool(b)

    def enable(self):
        self.set_enabled(True)

    def disable(self):
        self.set_enabled(False)

    def set_maxcount(self, count):
        self._maxcount = count

    def set_maxsize(self, size):
        self._maxsize = size

    def set_policy(self, policy):
        self._policy = policy

    def set(self, key, value):
        self.__setitem__(key, value)

    def get(self, key: str, default=object):
        try:
            return self.__getitem__(key)
        except KeyError:
            if default == object:
                raise
            return default

    def _check_replacement(self):
        if self._maxcount and self.count - 1 > self._maxcount:
            return True
        if self._maxsize and self.size > self._maxsize:
            return True
        return False

    def _replace(self):
        pass

    def replace_size(self, required_space=0):
        if self._maxsize:
            if required_space >= self._maxsize:
                self.clear()
            else:
                while self.size + required_space > self._maxsize:
                    self._replace()

    def replace_count(self):
        if self._maxcount:
            while self.count >= self._maxcount:
                self._replace()

    def replace(self, required_space=0):
        self.replace_size(required_space)
        self.replace_count()

    def __repr__(self):
        size = formatsize(self.size, dec=1, delim="")
        return f"{self.__class__.__name__}({self.count}, {size})"


class EigenCache(AbstractCache):

    FILE_PREFIX = "eig_"

    def __init__(self, root="", maxcount=0, maxsize=0, policy="FIFO", autosave=True):
        super().__init__(maxcount, maxsize, policy)
        self.__items__ = OrderedDict()
        self._autosave = autosave
        self.root = os.path.join(CACHE_DIR, root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def _replace(self):
        if self._policy in (self.LRU, self.FIFO):
            k = next(iter(self.__items__))
        else:
            k = random.choice(list(self.keys()))
        self.pop(k)

    def __sizeof__(self):
        return getsizeof(self.__items__)

    def __len__(self):
        return len(self.__items__)

    def __iter__(self):
        if not self._enabled:
            return None
        return iter(self.__items__)

    def __contains__(self, key):
        if key in self.__items__:
            return True
        return os.path.exists(os.path.join(self.root, self.FILE_PREFIX + key))

    def __getitem__(self, key):
        if not self._enabled:
            raise KeyError("Cache disabled!")
        try:
            val = self.__items__[key]
        except KeyError:
            self.load_item(key, raise_=False)
            val = self.__items__[key]

        if self._policy == self.LRU:
            self.__items__.move_to_end(key)
        return val

    def __setitem__(self, key, value):
        size = getsizeof(value)
        self.replace(size)

        self.__items__[key] = value
        if self._policy in (self.LRU, self.FIFO):
            self.__items__.move_to_end(key)

        if self._autosave:
            self.save_item(key)

    def __delitem__(self, key):
        del self.__items__[key]

    def __dict__(self):
        return dict(self.items())

    def clear(self) -> None:
        self.__items__.clear()
        if self._autosave:
            self.save()

    def save_item(self, key):
        file = os.path.join(self.root, self.FILE_PREFIX + key)
        eigvals, eigvecs = self.__getitem__(key)
        np.savez(file + ".npz", eigvals=eigvals, eigvecs=eigvecs)

    def load_item(self, key, raise_=True):
        name = self.FILE_PREFIX + key
        file = os.path.join(self.root, name)
        try:
            data = np.load(file + ".npz")
            eigvals = data["eigvals"]
            eigvecs = data["eigvecs"]
            self.__setitem__(key, (eigvals, eigvecs))
        except FileNotFoundError:
            if raise_:
                raise
        except Exception as e:
            logger.warning(f"Could not load file {name}: {e}")
            if raise_:
                raise

    def load_item_file(self, name, raise_=True):
        key = os.path.splitext(name.removeprefix(self.FILE_PREFIX))[0]
        self.load_item(key, raise_)

    def delete_item(self, key):
        file = os.path.join(self.root, self.FILE_PREFIX + key)
        os.remove(file)

    def delete_all(self):
        for name in list(os.listdir(self.root)):
            os.remove(os.path.join(self.root, name))

    def save(self):
        for key in self.keys():
            self.save_item(key)

    def load(self, raise_=True):
        self.clear()
        for name in os.listdir(self.root):
            if name.startswith("eig_"):
                self.load_item_file(name, raise_)


class EigenCacheHDF5(AbstractCache):
    def __init__(self, root="", mode="a", **kwargs):
        super().__init__(**kwargs)
        self.root = os.path.join(CACHE_DIR, root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.file = h5py.File(os.path.join(self.root, "eig.hdf5"), mode=mode)

    def __sizeof__(self):
        return getsizeof(self.file)

    def __len__(self):
        return len(self.file)

    def __iter__(self):
        return iter(self.file)

    def __setitem__(self, key, value):
        eigvals, eigvecs = value
        data = np.concatenate([eigvals[:, np.newaxis], eigvecs], axis=1)
        if key in self.file:
            dset = self.file[key]
            dset[:, :] = data
        else:
            self.file.create_dataset(key, data=data)

    def __getitem__(self, key):
        dset = self.file[key]
        eigvals = dset[:, 0]
        eigvecs = dset[:, 1:]
        return eigvals, eigvecs

    def __delitem__(self, key):
        del self.file[key]
