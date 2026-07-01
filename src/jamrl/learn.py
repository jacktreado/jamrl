"""Learner job: aggregate -> update -> write -> eval (plan 5.6/5.8).

Self-resubmission (step d) lives in slurm.py; this module performs the actual
PPO/CEM update and greedy evaluation for a single round.
"""
from __future__ import annotations

import os

import numpy as np

from jamrl import _core, cem as cem_mod, config, policy, ppo, rollout, seeding, storage
from jamrl.policy import OBS_DIM, Adam, PolicyNet, RunningNorm, ValueNet


# ----------------------------------------------------------------------- #
# Aggregation
# ----------------------------------------------------------------------- #
def aggregate_round(camp, r, workers):
    trajs, present = [], 0
    for k in range(workers):
        p = storage.rollout_path(camp, r, k)
        if os.path.exists(p):
            trajs.append(storage.read_rollout_npz(p))
            present += 1
    if not trajs:
        return None, 0

    def cat(key):
        return np.concatenate([t[key] for t in trajs])

    ep_ptr, off = [0], 0
    for t in trajs:
        for e in t["ep_ptr"][1:]:
            ep_ptr.append(off + int(e))
        off += int(t["ep_ptr"][-1])

    traj = {
        "obs": cat("obs"), "act": cat("act"), "rew": cat("rew"), "done": cat("done"),
        "ep_ptr": np.asarray(ep_ptr, np.int64),
        "phi": cat("phi"), "phi_null": cat("phi_null"),
        "outcome": cat("outcome"), "seeds": cat("seeds"), "steps": cat("steps"),
    }
    return traj, present


# ----------------------------------------------------------------------- #
# Checkpoint (NumPy backend)
# ----------------------------------------------------------------------- #
def _set_policy_from_npz(pol, d):
    pol.mlp.W = [np.asarray(d["W0"], float), np.asarray(d["W1"], float), np.asarray(d["Wmu"], float)]
    pol.mlp.b = [np.asarray(d["b0"], float), np.asarray(d["b1"], float), np.asarray(d["bmu"], float)]
    pol.log_std = np.asarray(d["log_std"], float)


def save_checkpoint(path, pol, val, norm, opt_p, opt_v, rnd):
    arrs = {
        "pW0": pol.mlp.W[0], "pb0": pol.mlp.b[0], "pW1": pol.mlp.W[1], "pb1": pol.mlp.b[1],
        "pWmu": pol.mlp.W[2], "pbmu": pol.mlp.b[2], "log_std": pol.log_std,
        "vW0": val.mlp.W[0], "vb0": val.mlp.b[0], "vW1": val.mlp.W[1], "vb1": val.mlp.b[1],
        "vWo": val.mlp.W[2], "vbo": val.mlp.b[2],
        "norm_mean": norm.mean, "norm_var": norm.var, "norm_count": np.float64(norm.count),
        "opt_p_t": np.int64(opt_p.t), "opt_v_t": np.int64(opt_v.t), "round": np.int64(rnd),
    }
    with storage.atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.savez(f, **arrs)


def load_checkpoint(path, pol, val, norm, opt_p, opt_v):
    with np.load(path) as z:
        pol.mlp.W = [z["pW0"], z["pW1"], z["pWmu"]]
        pol.mlp.b = [z["pb0"], z["pb1"], z["pbmu"]]
        pol.log_std = z["log_std"].copy()
        val.mlp.W = [z["vW0"], z["vW1"], z["vWo"]]
        val.mlp.b = [z["vb0"], z["vb1"], z["vbo"]]
        norm.load({"mean": z["norm_mean"], "var": z["norm_var"], "count": z["norm_count"]})
        opt_p.load({"t": int(z["opt_p_t"])})
        opt_v.load({"t": int(z["opt_v_t"])})


# ----------------------------------------------------------------------- #
# PPO round
# ----------------------------------------------------------------------- #
def _learn_ppo(cfg, camp, r, traj):
    from jamrl import torch_backend as tb

    if tb.resolve_backend(cfg) == "torch":
        return tb.learn_ppo_round(cfg, camp, r, traj)
    return _learn_ppo_numpy(cfg, camp, r, traj)


def _learn_ppo_numpy(cfg, camp, r, traj):
    pol = PolicyNet(hidden=cfg.hidden, act_dim=config.act_dim(cfg),
                    logstd_init=cfg.logstd_init, seed=cfg.seed)
    val = ValueNet(hidden=cfg.hidden, seed=cfg.seed + 1)
    norm = RunningNorm(OBS_DIM)
    opt_p, opt_v = Adam(cfg.lr), Adam(cfg.lr)

    ckpt = storage.checkpoint_path(camp, r)
    if os.path.exists(ckpt):
        load_checkpoint(ckpt, pol, val, norm, opt_p, opt_v)
    else:  # round 0: continue from the initial policy npz
        d = policy.load_policy_npz(storage.policy_path(camp, r))
        _set_policy_from_npz(pol, d)
        norm.mean = np.asarray(d["obs_mean"], float).copy()
        norm.var = np.maximum(np.asarray(d["obs_std"], float) ** 2 - 1e-8, 1e-8)

    stats = ppo.ppo_update(cfg, pol, val, norm, opt_p, opt_v, traj)
    policy.save_policy_npz(storage.policy_path(camp, r + 1), pol, norm)
    save_checkpoint(storage.checkpoint_path(camp, r + 1), pol, val, norm, opt_p, opt_v, r + 1)
    return stats


# ----------------------------------------------------------------------- #
# CEM round (self-contained candidate evaluation)
# ----------------------------------------------------------------------- #
def _eval_candidate_return(cfg, camp, template, flat, seeds, phin):
    pol_net = cem_mod.candidate_policy(template, flat)
    arrs = policy.policy_arrays(pol_net, RunningNorm(OBS_DIM))
    core_pol = policy.build_core_policy(arrs)
    proto = _core.make_system(cfg.N, int(seeds[0]), cfg.phi0, cfg.P)
    ec = config.env_config(cfg)
    sf = _core.SaveFlags(); sf.save_hessian = 0; sf.save_moduli = False; sf.save_contacts = False
    all_seeds, all_phin = [], []
    for s, pn in zip(seeds, phin):
        for _ in range(cfg.cem_eps_per_cand):
            all_seeds.append(int(s)); all_phin.append(float(pn))
    eps = _core.run_episodes_batch(proto, core_pol, all_seeds, ec, sf,
                                   config.parallel_mode_code(cfg), all_phin)
    return float(np.mean([float(np.asarray(e["rew"]).sum()) for e in eps]))


def _learn_cem(cfg, camp, r, traj):
    template = cem_mod.make_template(cfg)
    dim = cem_mod.flatten_policy(template).size

    state_path = storage.checkpoint_path(camp, r)
    cem = cem_mod.CEM(dim, cem_mod.flatten_policy(template), cfg.cem_sigma0, cfg.cem_elite_frac)
    if os.path.exists(state_path):
        with np.load(state_path) as z:
            if "cem_mu" in z:
                cem.load({"mu": z["cem_mu"], "sigma": z["cem_sigma"]})

    seeds = seeding.worker_seeds(cfg.seed, r, 0, max(2, cfg.episodes_per_worker))
    phin = rollout.ensure_null_cache(cfg, camp, seeds)
    rng = np.random.default_rng(seeding.episode_seed(cfg.seed, r, 999, 0) % (2**32))
    cands = cem.ask(cfg.cem_pop, rng)
    scores = np.array([_eval_candidate_return(cfg, camp, template, c, seeds, phin) for c in cands])
    elite_mean = cem.tell(cands, scores)

    # ship the distribution mean as the next policy
    mean_pol = cem_mod.candidate_policy(template, cem.mu)
    policy.save_policy_npz(storage.policy_path(camp, r + 1), mean_pol, RunningNorm(OBS_DIM))
    with storage.atomic_path(storage.checkpoint_path(camp, r + 1)) as tmp:
        with open(tmp, "wb") as f:
            np.savez(f, cem_mu=cem.mu, cem_sigma=cem.sigma, round=np.int64(r + 1))
    return {"mean_reward": float(scores.mean()), "elite_mean": elite_mean,
            "n_transitions": int(traj["ep_ptr"][-1]), "sigma_policy": float(cem.sigma.mean())}


# ----------------------------------------------------------------------- #
# Greedy evaluation
# ----------------------------------------------------------------------- #
def greedy_eval(cfg, camp, r) -> dict:
    d = dict(policy.load_policy_npz(storage.policy_path(camp, r)))
    d["log_std"] = np.full_like(np.asarray(d["log_std"], float), -100.0)  # deterministic
    core_pol = policy.build_core_policy(d)
    ec = config.env_config(cfg)
    sf = _core.SaveFlags(); sf.save_hessian = 0; sf.save_moduli = True; sf.save_contacts = False
    if cfg.reward_mode == "shear_modulus":
        # Paired lift vs the fixed null ensemble: eval on a subset of the ensemble
        # seeds and compare each episode's G to that seed's stored null G. The obs
        # is normalized by the broadcast G_mean (as in training), so pass that as gn.
        ens = rollout.ensure_null_ensemble(cfg, camp)
        n_eval = min(32, len(ens["jammed_seeds"]))
        seeds = [int(s) for s in ens["jammed_seeds"][:n_eval]]
        g_per_seed = [float(x) for x in ens["jammed_G"][:n_eval]]
        phin = [ens["phi_mean"]] * n_eval
        gn = [ens["G_mean"]] * n_eval
        cn = []
    else:
        seeds = [int(s) for s in cfg.eval_seeds]
        phin, gnull, cnull = rollout.ensure_null_baselines(cfg, camp, seeds)
        g_per_seed = None
        gn = [float(x) for x in gnull] if gnull is not None else []
        cn = [float(x) for x in cnull] if cnull is not None else []
    proto = _core.make_system(cfg.N, seeds[0], cfg.phi0, cfg.P)
    eps = _core.run_episodes_batch(proto, core_pol, seeds, ec, sf,
                                   config.parallel_mode_code(cfg), [float(x) for x in phin],
                                   [float(x) for x in gn], cn)

    jam = [e for e in eps if e["jammed"]]
    dphi = [e["phi"] - e["phi_null"] for e in eps]
    # eval_dG = paired shear-modulus lift over the null ensemble: G_agent(seed_i) -
    # G_null(seed_i), using the ensemble's stored per-seed null G.
    if g_per_seed is not None:
        dG = [eps[i]["G"] - g_per_seed[i] for i in range(len(eps))
              if eps[i]["jammed"] and "G" in eps[i]]
    else:
        dG = []
    # eval_speed = fractional force-eval speedup over the null protocol, on jammed
    # episodes (speed mode); >0 means the policy reaches jamming with less work.
    speedup = [(e["cost_null"] - e["cost_eval"]) / e["cost_null"]
               for e in jam if e.get("cost_null", 0.0) > 0.0]
    cost_eval_all = [e["cost_eval"] for e in eps if "cost_eval" in e]
    aP = [float(np.abs(np.clip(np.asarray(e["act"])[:, 0], -1, 1)).mean()) for e in eps if e["T"] > 0]
    aS = [float(np.abs(np.clip(np.asarray(e["act"])[:, 1], -1, 1)).mean()) for e in eps if e["T"] > 0]

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    return {
        "eval_dphi": mean(dphi),
        "eval_dG": mean(dG),
        "eval_speed": mean(speedup),
        "eval_cost_kevals": mean([c / 1000.0 for c in cost_eval_all]),
        "eval_success": len(jam) / len(eps) if eps else 0.0,
        "mean_absaP": mean(aP),
        "mean_absaS": mean(aS),
        "mean_absgamma": mean([abs(e["gamma"]) for e in jam]),
        "Bbar": mean([e["B"] for e in jam if "B" in e]),
        "Gbar": mean([e["G"] for e in jam if "G" in e]),
        "dzbar": mean([e["dz"] for e in jam]),
        "rattler_frac": mean([e["n_rattlers"] / cfg.N for e in jam]),
        "shear_stable_frac": mean([1.0 if e.get("G", 0.0) >= -1e-8 else 0.0 for e in jam]),
        # soft-mode edge of the jammed packings (lowest real VDOS eigenfreq)
        "omega_star": mean([e["omega_star"] for e in jam
                            if "omega_star" in e and np.isfinite(e["omega_star"])]),
    }


# ----------------------------------------------------------------------- #
def learn_round(cfg, camp, r) -> dict:
    import time
    t0 = time.time()
    traj, present = aggregate_round(camp, r, cfg.workers)
    if traj is None or present < cfg.min_worker_frac * cfg.workers:
        raise SystemExit(
            f"[learn] insufficient rollouts for round {r}: {present}/{cfg.workers} "
            f"(< min_worker_frac={cfg.min_worker_frac})"
        )

    if cfg.algo == "cem":
        stats = _learn_cem(cfg, camp, r, traj)
    else:
        stats = _learn_ppo(cfg, camp, r, traj)

    eval_stats = greedy_eval(cfg, camp, r + 1)
    prov_hash = ""
    prov = os.path.join(camp, "provenance.json")
    if os.path.exists(prov):
        import json
        with open(prov) as f:
            prov_hash = json.load(f).get("git_commit", "")[:12]

    row = {
        "round": r,
        "episodes": int(len(traj["ep_ptr"]) - 1),
        "mean_reward": stats.get("mean_reward", float("nan")),
        "sigma_policy": stats.get("sigma_policy", float("nan")),
        "wall_seconds": time.time() - t0,  # learn + eval wall time for this round
        "git_hash": prov_hash,
        **eval_stats,
    }
    storage.append_summary(camp, row)
    return {**stats, **eval_stats}
