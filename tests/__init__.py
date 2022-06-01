# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

from hypothesis import settings
from numba import config

config.DISABLE_JIT = True

settings.register_profile("exactdiag", deadline=None, report_multiple_bugs=True)
