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

from jamrl import _core, config, policy, seeding, staging, storage


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


def ensure_null_baselines(cfg, camp, seeds):
    """Return (phi_null, G_null|None) aligned with `seeds`, caching any missing.

    Density mode needs only phi_null (cheap; no Hessian). Shear mode also needs
    G_null (the null protocol's shear modulus at the same P); both are computed
    from a single null run via `_core.compute_null_phi_G` and cached separately.
    """
    if cfg.reward_mode != "shear_modulus":
        return ensure_null_cache(cfg, camp, seeds), None

    keys = [(cfg.N, cfg.P, int(s)) for s in seeds]
    cphi = storage.null_cache_get(camp, keys, field="phi")
    cG = storage.null_cache_get(camp, keys, field="G")
    missing = [k for k in keys if k not in cphi or k not in cG]
    if missing:
        ec = config.env_config(cfg)
        newphi, newG = {}, {}
        for (N, P, s) in missing:
            proto = _core.make_system(N, int(s), cfg.phi0, P)
            phi, G = _core.compute_null_phi_G(proto, ec)
            newphi[(N, P, s)] = float(phi)
            newG[(N, P, s)] = float(G)
        storage.null_cache_update(camp, newphi, field="phi")
        storage.null_cache_update(camp, newG, field="G")
        cphi.update(newphi)
        cG.update(newG)
    return [cphi[k] for k in keys], [cG[k] for k in keys]


def run_rollout(cfg, camp, r: int, k: int) -> list[dict]:
    """Execute one rollout array task; returns the episode dicts."""
    _core.set_num_threads(cfg.threads_per_task)

    pol_npz = policy.load_policy_npz(storage.policy_path(camp, r))
    pol = policy.build_core_policy(pol_npz)

    seeds = seeding.worker_seeds(cfg.seed, r, k, cfg.episodes_per_worker)
    phin, gnull = ensure_null_baselines(cfg, camp, seeds)

    proto = _core.make_system(cfg.N, int(seeds[0]) if seeds else 1, cfg.phi0, cfg.P)
    ec = config.env_config(cfg)
    sf = config.save_flags(cfg)
    pm = config.parallel_mode_code(cfg)

    gn = [float(x) for x in gnull] if gnull is not None else []
    episodes = _core.run_episodes_batch(
        proto, pol, [int(s) for s in seeds], ec, sf, pm, [float(x) for x in phin], gn
    )

    # Heavy outputs go to node-local scratch (if configured) and are copied to
    # the persistent campaign on completion; otherwise written in place.
    radii = np.asarray(proto.radii)
    with staging.output(storage.rollout_path(camp, r, k), cfg) as rp:
        storage.write_rollout_npz(rp, episodes)
    with staging.output(storage.states_path(camp, r, k), cfg) as sp:
        storage.write_states_h5(
            sp, episodes, radii, cfg.P,
            save_hessian=cfg.save_hessian, compression=cfg.compression,
        )
    return episodes
