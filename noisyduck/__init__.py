# -*- coding: utf-8 -*-

"""Top-level package for Noisy Duck."""

__author__ = """Nathan Wukie"""
__email__ = 'nathan.wukie@gmail.com'
__version__ = '0.2.0'

from . import annulus
from . import filter
from .io import write_decomposition

# silence flake8 PEP8 violation
__all__ = ['annulus', 'filter', 'write_decomposition']
