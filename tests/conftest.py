"""Make the in-tree package importable for tests without a full install.

Adds ``src/`` to ``sys.path`` so ``import jamrl`` / ``import jamrl._core``
resolve to the working tree (the locally built ``_core*.so`` is copied into
``src/jamrl/`` by ``scripts/build.sh``).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
