"""Rollout worker: run E episodes under the current policy, write data (plan 5.5).

For round r, worker k:
  1. load policy/round_r.npz -> _core.Policy
  2. ensure phi_null is cached for this worker's seeds (compute missing)
  3. derive episode seeds deterministically
  4. _core.run_episodes_batch(...)
  5. write rollouts/round_r/worker_k.npz and states/round_r/worker_k.h5
"""
from __future__ import annotations

import numpy as np

from jamrl import _core, config, policy, seeding, storage


def ensure_null_cache(cfg, camp, seeds) -> list[float]:
    """Return phi_null aligned with `seeds`, computing+caching any missing."""
    keys = [(cfg.N, cfg.P, int(s)) for s in seeds]
    cached = storage.null_cache_get(camp, keys)
    missing = [k for k in keys if k not in cached]
    if missing:
        ec = config.env_config(cfg)
        newvals = {}
        for (N, P, s) in missing:
            proto = _core.make_system(N, int(s), cfg.phi0, P)
            newvals[(N, P, s)] = float(_core.compute_null_phi(proto, ec))
        storage.null_cache_update(camp, newvals)
        cached.update(newvals)
    return [cached[k] for k in keys]


def run_rollout(cfg, camp, r: int, k: int) -> list[dict]:
    """Execute one rollout array task; returns the episode dicts."""
    _core.set_num_threads(cfg.threads_per_task)

    pol_npz = policy.load_policy_npz(storage.policy_path(camp, r))
    pol = policy.build_core_policy(pol_npz)

    seeds = seeding.worker_seeds(cfg.seed, r, k, cfg.episodes_per_worker)
    phin = ensure_null_cache(cfg, camp, seeds)

    proto = _core.make_system(cfg.N, int(seeds[0]) if seeds else 1, cfg.phi0, cfg.P)
    ec = config.env_config(cfg)
    sf = config.save_flags(cfg)
    pm = config.parallel_mode_code(cfg)

    episodes = _core.run_episodes_batch(
        proto, pol, [int(s) for s in seeds], ec, sf, pm, [float(x) for x in phin]
    )

    storage.write_rollout_npz(storage.rollout_path(camp, r, k), episodes)
    radii = np.asarray(proto.radii)
    storage.write_states_h5(
        storage.states_path(camp, r, k), episodes, radii, cfg.P,
        save_hessian=cfg.save_hessian, compression=cfg.compression,
    )
    return episodes
