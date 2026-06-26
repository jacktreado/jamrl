"""Phase 7 gate: config / seeding / storage / rollout worker write valid data."""
import os

import numpy as np
import pytest

from jamrl import config, policy, rollout, seeding, storage
from jamrl.config import Config


def test_config_yaml_and_args_roundtrip(tmp_path):
    c = Config(N=64, P=2e-3, hidden=(32, 32), eval_seeds=(1, 2, 3))
    p = tmp_path / "c.yaml"
    c.save_yaml(p)
    c2 = Config.from_yaml(p)
    assert c2.N == 64 and c2.P == 2e-3
    assert c2.hidden == (32, 32) and c2.eval_seeds == (1, 2, 3)
    assert c.config_hash() == c2.config_hash()


def test_config_from_args_precedence(tmp_path):
    import argparse

    base = Config(N=128, w_phi=100.0)
    yml = tmp_path / "b.yaml"
    base.save_yaml(yml)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    config.add_arguments(parser)
    args = parser.parse_args(["--config", str(yml), "--N", "256", "--hidden", "16,16"])
    cfg = config.from_args(args)
    assert cfg.N == 256          # CLI overrides YAML
    assert cfg.w_phi == 100.0    # YAML value retained
    assert cfg.hidden == (16, 16)


def test_seeding_deterministic_and_distinct():
    a = seeding.episode_seed(12345, 0, 0, 0)
    b = seeding.episode_seed(12345, 0, 0, 0)
    c = seeding.episode_seed(12345, 0, 0, 1)
    assert a == b and a != c
    ws = seeding.worker_seeds(12345, 1, 2, 5)
    assert len(ws) == 5 and len(set(ws)) == 5


def test_policy_npz_roundtrip_and_core_build(tmp_path):
    p = tmp_path / "round_0000.npz"
    pol, norm = policy.init_policy_npz(str(p), hidden=(16, 16), seed=0)
    d = policy.load_policy_npz(str(p))
    core_pol = policy.build_core_policy(d)
    o = np.zeros(10)
    mu = core_pol.forward(o)
    assert mu.shape == (2,)
    # numpy net and C++ net agree on the mean for zero obs
    np_mu = pol.mu(norm.normalize(o[None]))[0]
    assert np.allclose(mu, np_mu, atol=1e-10)


def test_rollout_worker_writes_valid_data(tmp_path):
    cfg = Config(N=64, P=1e-3, phi0=0.80, T_cap=20, episodes_per_worker=3,
                 threads_per_task=2, save_hessian="sparse",
                 campaign_root=str(tmp_path), name="smoke")
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden,
                           logstd_init=cfg.logstd_init, seed=cfg.seed)

    episodes = rollout.run_rollout(cfg, camp, 0, 0)
    assert len(episodes) == 3

    # trajectory npz schema
    data = storage.read_rollout_npz(storage.rollout_path(camp, 0, 0))
    E = cfg.episodes_per_worker
    assert data["obs"].shape[1] == 10 and data["act"].shape[1] == 2
    assert data["ep_ptr"].shape == (E + 1,)
    assert data["ep_ptr"][-1] == data["obs"].shape[0] == data["rew"].shape[0]
    assert data["seeds"].shape == (E,) and data["outcome"].shape == (E,)
    assert data["phi"].shape == (E,) and data["phi_null"].shape == (E,)
    assert data["done"].sum() == E  # one terminal per episode

    # null cache populated and reused (no recompute)
    assert os.path.exists(storage.null_cache_path(camp))
    seeds = seeding.worker_seeds(cfg.seed, 0, 0, E)
    cached = storage.null_cache_get(camp, [(cfg.N, cfg.P, int(s)) for s in seeds])
    assert len(cached) == E

    # jammed-state h5
    n_jammed = sum(1 for e in episodes if e["jammed"])
    groups = list(storage.iter_states_h5(storage.states_path(camp, 0, 0)))
    assert len(groups) == n_jammed
    if groups:
        name, attrs, g = groups[0]
        for key in ("seed", "L", "gamma", "P_target", "P_int", "phi", "z", "dz",
                    "n_keep", "n_rattlers", "n_contacts"):
            assert key in attrs
        assert g["s"].shape == (cfg.N, 2)
        assert g["radii"].shape == (cfg.N,)
        assert g["contacts"].shape[1] == 2
        assert "H_data" in g and "H_indices" in g and "H_indptr" in g
