"""Phase 6 gate: batch runner parity + episode-parallel reproducibility (plan 8)."""
import numpy as np
import pytest

import jamrl._core as core


def make_policy(obs_dim=16, hidden=(16, 16), act_dim=2, seed=0, scale=0.2):  # obs_dim = jamcore::OBS_DIM
    rng = np.random.default_rng(seed)
    obs_mean = np.zeros(obs_dim)
    obs_std = np.ones(obs_dim)
    W0 = rng.standard_normal((hidden[0], obs_dim)) * scale
    b0 = np.zeros(hidden[0])
    W1 = rng.standard_normal((hidden[1], hidden[0])) * scale
    b1 = np.zeros(hidden[1])
    Wmu = rng.standard_normal((act_dim, hidden[1])) * scale
    bmu = np.zeros(act_dim)
    log_std = np.full(act_dim, -0.5)
    return core.Policy(obs_mean, obs_std, W0, b0, W1, b1, Wmu, bmu, log_std)


def cfg_for(phi0=0.80):
    cfg = core.EnvConfig()
    cfg.phi0 = phi0
    return cfg


def test_batch_parity_with_step_loop():
    """One batch episode == a manual Env/Policy/Rng step loop, bit for bit."""
    N, P, phi0, seed = 32, 1e-3, 0.80, 24681
    cfg = cfg_for(phi0)
    pol = make_policy()
    proto = core.make_system(N, seed, phi0, P)
    phin = core.compute_null_phi(core.make_system(N, seed, phi0, P), cfg)

    res = core.run_episodes_batch(proto, pol, [seed], cfg, core.SaveFlags(), 0, [phin])
    e = res[0]

    # manual replication using the same C++ primitives
    sys = core.make_system(N, seed, phi0, P)
    env = core.Env()
    env.cfg = cfg
    obs = env.reset(sys, phin)
    rng = core.Rng(core.action_subseed(seed))
    obsL, actL, rewL = [], [], []
    while not env.done:
        a = pol.sample(obs, rng)
        tr = env.step(a[0], a[1])
        obsL.append(np.asarray(obs))
        actL.append(np.asarray(a))
        rewL.append(tr["reward"])
        obs = tr["obs"]

    assert np.array_equal(np.asarray(e["obs"]), np.array(obsL))
    assert np.array_equal(np.asarray(e["act"]), np.array(actL))
    assert np.array_equal(np.asarray(e["rew"]), np.array(rewL))
    assert e["outcome"] == env.outcome
    assert e["T"] == len(obsL)


def test_episode_parallel_reproducible():
    """Shuffling the seed order does not change per-seed results."""
    N, P, phi0 = 32, 1e-3, 0.80
    cfg = cfg_for(phi0)
    pol = make_policy(seed=1)
    proto = core.make_system(N, 1, phi0, P)
    seeds = [101, 202, 303, 404, 505, 606]
    phin = [core.compute_null_phi(core.make_system(N, s, phi0, P), cfg) for s in seeds]

    res = core.run_episodes_batch(proto, pol, seeds, cfg, core.SaveFlags(), 0, phin)
    by_seed = {int(e["seed"]): e for e in res}

    order = [505, 101, 606, 303, 202, 404]
    idx = [seeds.index(s) for s in order]
    res2 = core.run_episodes_batch(proto, pol, order, cfg, core.SaveFlags(), 0, [phin[i] for i in idx])
    by_seed2 = {int(e["seed"]): e for e in res2}

    for s in seeds:
        a, b = by_seed[s], by_seed2[s]
        assert np.array_equal(np.asarray(a["obs"]), np.asarray(b["obs"])), f"seed {s}"
        assert np.array_equal(np.asarray(a["rew"]), np.asarray(b["rew"])), f"seed {s}"
        assert a["outcome"] == b["outcome"]
        assert a["phi"] == b["phi"]


def test_jammed_state_payload():
    """Jammed episodes carry geometry, contacts, moduli, and (opt) Hessian."""
    N, P, phi0 = 32, 1e-3, 0.80
    cfg = cfg_for(phi0)
    pol = make_policy(seed=3, scale=0.05)  # gentle policy -> tends to jam
    save = core.SaveFlags()
    save.save_moduli = True
    save.save_hessian = 2  # sparse
    seeds = [11, 12, 13, 14, 15, 16]
    proto = core.make_system(N, 1, phi0, P)
    res = core.run_episodes_batch(proto, pol, seeds, cfg, save, 0, [])
    jammed = [e for e in res if e["jammed"]]
    assert len(jammed) >= 1
    e = jammed[0]
    assert e["x_final"].shape[0] == 2 * N + 2
    assert 0.5 < e["phi"] < 0.95
    assert e["contacts"].shape[1] == 2
    assert "B" in e and "G" in e and e["G"] >= -1e-8
    assert "H_data" in e and tuple(e["H_shape"]) == (2 * N + 2, 2 * N + 2)
    assert e["n_keep"] + e["n_rattlers"] == N


def test_batch_threads_match_serial():
    """parallel_mode=0 with many threads == single-thread results (determinism)."""
    N, P, phi0 = 32, 1e-3, 0.80
    cfg = cfg_for(phi0)
    pol = make_policy(seed=7)
    seeds = list(range(50, 66))
    proto = core.make_system(N, 1, phi0, P)
    core.set_num_threads(1)
    r1 = core.run_episodes_batch(proto, pol, seeds, cfg, core.SaveFlags(), 0, [])
    core.set_num_threads(max(2, core.max_threads()))
    r2 = core.run_episodes_batch(proto, pol, seeds, cfg, core.SaveFlags(), 0, [])
    for a, b in zip(r1, r2):
        assert np.array_equal(np.asarray(a["rew"]), np.asarray(b["rew"]))
        assert a["outcome"] == b["outcome"]
