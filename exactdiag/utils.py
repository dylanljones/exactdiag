# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import os
import tomli
import numba
import logging

logger = logging.getLogger(__name__.split(".")[0])

# Logging format
frmt = "[%(asctime)s] %(name)s:%(levelname)-8s - %(message)s"
formatter = logging.Formatter(frmt, datefmt="%H:%M:%S")

# Set up console logger
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

# Set logging level
logger.setLevel(logging.WARNING)
logging.root.setLevel(logging.NOTSET)


def _read_config():
    files = ["ed.toml", "exactdiag.toml", "pyproject.toml"]
    for file in files:
        try:
            with open(file, "rb") as fh:
                data = tomli.load(fh)
            return dict(data["exactdiag"])
        except (FileNotFoundError, KeyError):
            pass
    return dict()


def parse_config():

    data = _read_config()
    num_threads = data.get("numba_threads", -1)
    if num_threads <= 0:
        num_threads = os.cpu_count() + num_threads

    cache_dir = data.get("cache_dir", "")
    if not cache_dir:
        cache_dir = "__edcache__"

    return {
        "numba_threads": num_threads,
        "cache_dir": cache_dir,
    }


CONFIG = parse_config()
CACHE_DIR = CONFIG["cache_dir"]

numba.set_num_threads(CONFIG["numba_threads"])
