"""Simulate and sample ecological communities across space and time."""
from importlib.metadata import version, PackageNotFoundError

__version__ = version('lovebirds')

from .main import fEnv, Species, Sim, run_demo

# Public API exports will go here as the package stabilizes.
