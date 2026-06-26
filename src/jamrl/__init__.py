"""jamrl — distributed RL of box-control jamming protocols.

The compiled physics core lives in :mod:`jamrl._core` (built from ``cpp/``).
Importing this package is intentionally lightweight: heavy optional
dependencies (torch, h5py, pandas, matplotlib) are imported lazily inside the
modules that need them, so ``import jamrl._core`` never drags them in.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
