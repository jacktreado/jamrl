"""Rollout worker: run E episodes under the current policy, write data (plan 5.5).

For round r, worker k:
  1. load policy/round_r.npz -> _core.Policy
  2. ensure phi_null is cached for this worker's seeds (compute missing)
  3. derive episode seeds deterministically
  4. _core.run_episodes_batch(...)
  5. write rollouts/round_r/worker_k.npz and states/round_r/worker_k.h5
"""
from __future__ import annotations

import json
import os

import numpy as np

from jamrl import _core, config, policy, seeding, staging, storage

# Reserved "round" namespace for the per-campaign null ensemble seeds, disjoint
# from training (round in [0, rounds)) and eval (small int) seeds.
_NULL_ENSEMBLE_ROUND = 0xE55E_5EED


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


# --------------------------------------------------------------------------- #
# Per-campaign null ensemble (fixed shear-reward baseline; plan section C).
# --------------------------------------------------------------------------- #
def null_ensemble_seeds(cfg, n: int) -> list[int]:
    """Deterministic reserved seeds for the campaign's null ensemble."""
    return [seeding.episode_seed(cfg.seed, _NULL_ENSEMBLE_ROUND, 0, i) for i in range(n)]


def _zero_action_policy(cfg):
    """A deterministic ~zero-action policy (zero weights, vanishing std) so the
    batch runner reproduces the zero-action null protocol under OpenMP."""
    h0, h1 = cfg.hidden
    Z = lambda *s: np.zeros(s, np.float64)  # noqa: E731
    d = policy.OBS_DIM
    return _core.Policy(Z(d), np.ones(d), Z(h0, d), Z(h0), Z(h1, h0), Z(h1),
                        Z(policy.ACT_DIM, h1), Z(policy.ACT_DIM), np.full(policy.ACT_DIM, -100.0))


def load_null_ensemble(camp) -> dict:
    with open(storage.null_ensemble_path(camp)) as f:
        return json.load(f)


def compute_null_ensemble(cfg, camp, n_null: int | None = None) -> dict:
    """Generate a fixed ensemble of `n_null` zero-action null packings (same
    N/P/phi0, reserved held-out seeds) and persist its baseline stats to
    null_ensemble.json. Its mean G is the campaign's shear-reward reference.
    Idempotent + cross-process safe; returns the loaded ensemble dict."""
    path = storage.null_ensemble_path(camp)
    if os.path.exists(path):
        return load_null_ensemble(camp)
    with storage.file_lock(path):
        if os.path.exists(path):  # another worker won the race
            return load_null_ensemble(camp)
        n = int(n_null if n_null is not None else cfg.n_null)
        seeds = null_ensemble_seeds(cfg, n)
        # Run all n zero-action episodes in parallel (OpenMP over episodes).
        # DENSITY mode + a dummy phi_null avoids any internal per-seed null
        # recompute; vdos_obs off keeps per-step cost minimal. out.G is the
        # shear modulus at the jammed terminal (save_moduli).
        _core.set_num_threads(max(1, cfg.threads_per_task))
        ec = config.env_config(cfg)
        ec.reward_mode = 0
        ec.vdos_obs = False
        proto = _core.make_system(cfg.N, int(seeds[0]), cfg.phi0, cfg.P)
        sf = _core.SaveFlags()
        sf.save_hessian = 0
        sf.save_moduli = True
        sf.save_contacts = False
        eps = _core.run_episodes_batch(proto, _zero_action_policy(cfg), [int(s) for s in seeds],
                                       ec, sf, config.parallel_mode_code(cfg), [1.0] * n, [], [])
        js, jG, jphi = [], [], []
        for s, e in zip(seeds, eps):
            if not e.get("jammed"):
                continue
            G = float(e.get("G", float("nan")))
            phi = float(e.get("phi", float("nan")))
            if np.isfinite(G) and G > 0.0 and np.isfinite(phi) and phi > 0.0:
                js.append(int(s)); jG.append(G); jphi.append(phi)
        if not jG:
            raise RuntimeError(
                f"null ensemble: 0/{n} packings jammed at N={cfg.N} P={cfg.P}; "
                f"raise finish_cap_max or revisit the regime.")
        G_arr, phi_arr = np.asarray(jG), np.asarray(jphi)
        ens = {
            "N": cfg.N, "P": cfg.P, "phi0": cfg.phi0, "reward_mode": cfg.reward_mode,
            "n_null": n, "n_jammed": len(jG),
            "G_mean": float(G_arr.mean()), "G_std": float(G_arr.std()),
            "G_median": float(np.median(G_arr)),
            "phi_mean": float(phi_arr.mean()), "phi_std": float(phi_arr.std()),
            "jammed_seeds": js, "jammed_G": jG, "jammed_phi": jphi,
        }
        with storage.atomic_path(path) as tmp:
            with open(tmp, "w") as f:
                json.dump(ens, f, indent=2)
    return load_null_ensemble(camp)


def ensure_null_ensemble(cfg, camp) -> dict | None:
    """Compute the campaign null ensemble if missing (shear mode only)."""
    if cfg.reward_mode != "shear_modulus":
        return None
    return compute_null_ensemble(cfg, camp)


def run_rollout(cfg, camp, r: int, k: int) -> list[dict]:
    """Execute one rollout array task; returns the episode dicts."""
    _core.set_num_threads(cfg.threads_per_task)

    pol_npz = policy.load_policy_npz(storage.policy_path(camp, r))
    pol = policy.build_core_policy(pol_npz)

    seeds = seeding.worker_seeds(cfg.seed, r, k, cfg.episodes_per_worker)
    if cfg.reward_mode == "shear_modulus":
        # Fixed campaign baseline: broadcast the ensemble mean G (and phi) to
        # every episode -> reward w_G*(G/G_mean - 1); no per-seed null runs.
        ens = ensure_null_ensemble(cfg, camp)
        phin = [ens["phi_mean"]] * len(seeds)
        gnull = [ens["G_mean"]] * len(seeds)
        cnull = None
    else:
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
