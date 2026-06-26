"""Deterministic per-episode seed derivation + run provenance (plan section 5.2).

Seeds are derived with the same splitmix64-style mixer the C++ core uses
(``_core.mix_seed``), so a seed produced here reproduces a packing bit-for-bit
in the engine.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time

from jamrl import _core


def episode_seed(campaign_seed: int, rnd: int, worker: int, ep_idx: int) -> int:
    """64-bit per-episode seed from (campaign_seed, round, worker, episode)."""
    return int(_core.mix_seed(int(campaign_seed) & 0xFFFFFFFFFFFFFFFF,
                              int(rnd) & 0xFFFFFFFFFFFFFFFF,
                              int(worker) & 0xFFFFFFFFFFFFFFFF,
                              int(ep_idx) & 0xFFFFFFFFFFFFFFFF))


def worker_seeds(campaign_seed: int, rnd: int, worker: int, n: int) -> list[int]:
    """The n episode seeds a given (round, worker) will use."""
    return [episode_seed(campaign_seed, rnd, worker, i) for i in range(n)]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "absent"


def write_provenance(campaign_dir: str, config) -> dict:
    """Record git/commit, config hash, host, versions, env -> provenance.json."""
    prov = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "config_hash": config.config_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "torch": _pkg_version("torch"),
        "core_version": getattr(_core, "__version__", "?"),
        "core_openmp": bool(_core.has_openmp()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "campaign_seed": config.seed,
    }
    os.makedirs(campaign_dir, exist_ok=True)
    with open(os.path.join(campaign_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return prov
