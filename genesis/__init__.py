"""GENESIS — an artificial evolving universe."""
from .config import WorldConfig
from .engine.simulation import Simulation

__all__ = ["WorldConfig", "Simulation"]
__version__ = "0.1.0"
