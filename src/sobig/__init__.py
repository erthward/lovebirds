"""Simulation of observations of biodiversity across gradients."""
from importlib.metadata import version, PackageNotFoundError

__version__ = version('sobig')

from .main import fEnv, Species, Sim, run_demo

# Public API exports will go here as the package stabilizes.
