# __init__.py

# Import key classes and functions
from .core import SCHiCGenerator, generate_hic_file
from .utils import rlencode, create_mask

# Package metadata
__version__ = '0.1'
__author__ = 'Yun-Shuo Chou'
__email__ = 'yunshuoc@andrew.cmu.edu'
__description__ = 'A package to process Higashi output and generate HDF5 data for the CellScope tool.'

# Initialization logic (if any)
# For example, setting up logging, configuration, etc.
# import logging
# logging.basicConfig(level=logging.INFO)

# Optional: Define what gets imported with 'from package import *'
__all__ = ['SCHiCGenerator', 'generate_hic_file', 'rlencode', 'create_mask']