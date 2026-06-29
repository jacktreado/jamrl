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
    """Return (phi_null, G_null|None, cost_null|None) aligned with `seeds`, caching missing.

    Density mode needs only phi_null (cheap; no Hessian). Shear mode also needs
    G_null (the null protocol's shear modulus at the same P). Speed mode needs
    cost_null (the null protocol's force-eval count). The extra baselines are
    computed from a single null run via `_core.compute_null_baselines` and cached
    in separate per-field shards.
    """
    if cfg.reward_mode == "shear_modulus":
        fields = ("G",)
    elif cfg.reward_mode == "speed":
        fields = ("cost",)
    else:
        return ensure_null_cache(cfg, camp, seeds), None, None

    keys = [(cfg.N, cfg.P, int(s)) for s in seeds]
    cphi = storage.null_cache_get(camp, keys, field="phi")
    cextra = {fld: storage.null_cache_get(camp, keys, field=fld) for fld in fields}
    missing = [k for k in keys if k not in cphi or any(k not in cextra[fld] for fld in fields)]
    if missing:
        ec = config.env_config(cfg)
        newphi = {}
        newextra = {fld: {} for fld in fields}
        for (N, P, s) in missing:
            proto = _core.make_system(N, int(s), cfg.phi0, P)
            nb = _core.compute_null_baselines(proto, ec)  # {phi, G, cost}
            newphi[(N, P, s)] = float(nb["phi"])
            for fld in fields:
                newextra[fld][(N, P, s)] = float(nb[fld])
        storage.null_cache_update(camp, newphi, field="phi")
        cphi.update(newphi)
        for fld in fields:
            storage.null_cache_update(camp, newextra[fld], field=fld)
            cextra[fld].update(newextra[fld])

    phin = [cphi[k] for k in keys]
    if cfg.reward_mode == "shear_modulus":
        return phin, [cextra["G"][k] for k in keys], None
    return phin, None, [cextra["cost"][k] for k in keys]


def run_rollout(cfg, camp, r: int, k: int) -> list[dict]:
    """Execute one rollout array task; returns the episode dicts."""
    _core.set_num_threads(cfg.threads_per_task)

    pol_npz = policy.load_policy_npz(storage.policy_path(camp, r))
    pol = policy.build_core_policy(pol_npz)

    seeds = seeding.worker_seeds(cfg.seed, r, k, cfg.episodes_per_worker)
    phin, gnull, cnull = ensure_null_baselines(cfg, camp, seeds)

    proto = _core.make_system(cfg.N, int(seeds[0]) if seeds else 1, cfg.phi0, cfg.P)
    ec = config.env_config(cfg)
    sf = config.save_flags(cfg)
    pm = config.parallel_mode_code(cfg)

    gn = [float(x) for x in gnull] if gnull is not None else []
    cn = [float(x) for x in cnull] if cnull is not None else []
    episodes = _core.run_episodes_batch(
        proto, pol, [int(s) for s in seeds], ec, sf, pm, [float(x) for x in phin], gn, cn
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
