"""Gate: reward modes (density vs shear_modulus) — plan: multiple reward objectives."""
import argparse
import os

import numpy as np
import pytest

import jamrl._core as core
from jamrl import config, learn, policy, rollout, storage
from jamrl.config import Config


# --------------------------------------------------------------------------- #
# 1. Config plumbing
# --------------------------------------------------------------------------- #
def test_reward_mode_cli_and_env_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    config.add_arguments(parser)
    args = parser.parse_args(["--reward-mode", "shear_modulus", "--w-G", "123"])
    cfg = config.from_args(args)
    assert cfg.reward_mode == "shear_modulus"
    assert cfg.w_G == 123.0

    ec = config.env_config(cfg)
    assert ec.reward_mode == 1   # _REWARD_MODE_CODE["shear_modulus"]
    assert ec.w_G == 123.0

    # density default maps to code 0
    assert config.env_config(Config()).reward_mode == 0


def test_unknown_reward_mode_raises():
    with pytest.raises(ValueError):
        config.env_config(Config(reward_mode="bogus"))


# --------------------------------------------------------------------------- #
# 2. Null baseline: compute_null_phi_G returns finite phi (>0) and G; phi
#    matches the density-only null computation.
# --------------------------------------------------------------------------- #
def test_compute_null_phi_G_matches_phi():
    cfg = Config(N=32, P=1e-3, phi0=0.80)
    ec = config.env_config(cfg)
    proto = core.make_system(cfg.N, 7, cfg.phi0, cfg.P)

    phi_only = float(core.compute_null_phi(proto, ec))
    phi, G = core.compute_null_phi_G(proto, ec)
    assert np.isfinite(phi) and phi > 0.0
    assert np.isfinite(G)
    assert phi == pytest.approx(phi_only, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. Reward differs by mode. The trajectory is identical across modes (reward
#    doesn't feed back into action sampling), so only the terminal objective
#    term differs: rew_shear[-1] - rew_density[-1] == w_G*(G-G_null) - w_phi*(phi-phi_null).
# --------------------------------------------------------------------------- #
def test_reward_term_swaps_with_mode():
    N, w_G = 32, 200.0
    seeds = [1, 2, 3, 4]
    dcfg = Config(N=N, P=1e-3, phi0=0.80, reward_mode="density")
    scfg = Config(N=N, P=1e-3, phi0=0.80, reward_mode="shear_modulus", w_G=w_G)

    pol = policy.build_core_policy(
        policy.policy_arrays(*_fresh_policy())
    )
    proto = core.make_system(N, seeds[0], 0.80, 1e-3)
    ec_d = config.env_config(dcfg)
    ec_s = config.env_config(scfg)

    # baselines per seed (one null run yields both)
    phin, gnull = [], []
    for s in seeds:
        p = core.make_system(N, s, 0.80, 1e-3)
        phi, G = core.compute_null_phi_G(p, ec_s)
        phin.append(float(phi)); gnull.append(float(G))

    sf = core.SaveFlags(); sf.save_hessian = 0; sf.save_moduli = True; sf.save_contacts = False
    eps_d = core.run_episodes_batch(proto, pol, seeds, ec_d, sf, 1, phin, [])
    eps_s = core.run_episodes_batch(proto, pol, seeds, ec_s, sf, 1, phin, gnull)

    compared = 0
    for i, (ed, es) in enumerate(zip(eps_d, eps_s)):
        # identical trajectories
        assert np.allclose(np.asarray(ed["act"]), np.asarray(es["act"]))
        if not (ed["jammed"] and es["jammed"] and "G" in es):
            continue
        delta = float(es["rew"][-1]) - float(ed["rew"][-1])
        expected = w_G * (es["G"] - gnull[i]) - dcfg.w_phi * (es["phi"] - phin[i])
        assert delta == pytest.approx(expected, rel=1e-6, abs=1e-6)
        compared += 1
    assert compared > 0  # at least one jammed episode actually exercised the swap


def _fresh_policy():
    pn = policy.PolicyNet(hidden=(16, 16), seed=0)
    nm = policy.RunningNorm(policy.OBS_DIM)
    return pn, nm


# --------------------------------------------------------------------------- #
# 4. Back-compat: an old config.yaml lacking reward_mode loads as density.
# --------------------------------------------------------------------------- #
def test_old_config_defaults_to_density(tmp_path):
    d = Config(N=64).to_dict()
    d.pop("reward_mode", None)
    d.pop("w_G", None)
    cfg = Config.from_dict(d)
    assert cfg.reward_mode == "density"
    assert cfg.w_G == Config().w_G
    assert config.env_config(cfg).reward_mode == 0


# --------------------------------------------------------------------------- #
# 5. eval_dG: greedy_eval in shear mode returns a finite eval_dG; storage
#    round-trips the new summary column.
# --------------------------------------------------------------------------- #
def test_greedy_eval_emits_eval_dG(tmp_path):
    cfg = Config(N=32, P=1e-3, phi0=0.80, reward_mode="shear_modulus",
                 hidden=(16, 16), eval_seeds=(101, 102, 103),
                 campaign_root=str(tmp_path), name="dG", seed=1)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)

    stats = learn.greedy_eval(cfg, camp, 0)
    assert "eval_dG" in stats
    assert np.isfinite(stats["eval_dG"])


def test_summary_roundtrips_eval_dG(tmp_path):
    cfg = Config(campaign_root=str(tmp_path), name="rt")
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    storage.append_summary(camp, {"round": 0, "eval_dphi": 0.001, "eval_dG": 0.5})
    df = storage.read_summary(camp)
    assert "eval_dG" in df.columns
    assert float(df.iloc[0]["eval_dG"]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 6. Speed reward mode (force-eval cost, gated on the null density floor).
# --------------------------------------------------------------------------- #
def test_speed_mode_cli_and_env_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    config.add_arguments(parser)
    args = parser.parse_args(["--reward-mode", "speed", "--w-speed", "321"])
    cfg = config.from_args(args)
    assert cfg.reward_mode == "speed"
    assert cfg.w_speed == 321.0
    ec = config.env_config(cfg)
    assert ec.reward_mode == 2  # _REWARD_MODE_CODE["speed"]
    assert ec.w_speed == 321.0


def test_compute_null_baselines_consistent():
    cfg = Config(N=32, P=1e-3, phi0=0.80)
    ec = config.env_config(cfg)
    proto = core.make_system(cfg.N, 7, cfg.phi0, cfg.P)
    nb = core.compute_null_baselines(proto, ec)
    phi_only = float(core.compute_null_phi(proto, ec))
    assert nb["phi"] == pytest.approx(phi_only, rel=1e-9, abs=1e-12)
    assert np.isfinite(nb["G"])
    assert nb["cost"] > 0.0  # null episode does real relaxation work


def test_speed_reward_term_and_cost_recorded():
    """Speed mode records per-episode force-eval cost and, on jammed episodes that
    meet the null density floor, swaps the density term for w_speed*(cost_null-cost)/cost_null."""
    N, w_speed = 32, 200.0
    seeds = [1, 2, 3, 4]
    dcfg = Config(N=N, P=1e-3, phi0=0.80, reward_mode="density")
    pcfg = Config(N=N, P=1e-3, phi0=0.80, reward_mode="speed", w_speed=w_speed)
    pol = policy.build_core_policy(policy.policy_arrays(*_fresh_policy()))
    proto = core.make_system(N, seeds[0], 0.80, 1e-3)
    ec_d = config.env_config(dcfg)
    ec_p = config.env_config(pcfg)

    phin, cnull = [], []
    for s in seeds:
        p = core.make_system(N, s, 0.80, 1e-3)
        nb = core.compute_null_baselines(p, ec_p)
        phin.append(float(nb["phi"])); cnull.append(float(nb["cost"]))

    sf = core.SaveFlags(); sf.save_hessian = 0; sf.save_moduli = False; sf.save_contacts = False
    eps_d = core.run_episodes_batch(proto, pol, seeds, ec_d, sf, 1, phin, [], [])
    eps_p = core.run_episodes_batch(proto, pol, seeds, ec_p, sf, 1, phin, [], cnull)

    compared = 0
    for i, (ed, ep) in enumerate(zip(eps_d, eps_p)):
        assert np.allclose(np.asarray(ed["act"]), np.asarray(ep["act"]))  # identical trajectory
        assert ep["cost_eval"] > 0.0
        assert ep["cost_null"] == pytest.approx(cnull[i], rel=1e-9)
        if not (ed["jammed"] and ep["jammed"]):
            continue
        # terminal-term swap only when the density floor is met
        if ep["phi"] >= phin[i]:
            delta = float(ep["rew"][-1]) - float(ed["rew"][-1])
            expected = (w_speed * (cnull[i] - ep["cost_eval"]) / cnull[i]
                        - dcfg.w_phi * (ep["phi"] - phin[i]))
            assert delta == pytest.approx(expected, rel=1e-6, abs=1e-6)
            compared += 1
    assert compared > 0


def test_greedy_eval_emits_eval_speed(tmp_path):
    cfg = Config(N=32, P=1e-3, phi0=0.80, reward_mode="speed",
                 hidden=(16, 16), eval_seeds=(101, 102, 103),
                 campaign_root=str(tmp_path), name="spd", seed=1)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)

    stats = learn.greedy_eval(cfg, camp, 0)
    assert "eval_speed" in stats and "eval_cost_kevals" in stats
    assert np.isfinite(stats["eval_cost_kevals"])


def test_summary_roundtrips_eval_speed(tmp_path):
    cfg = Config(campaign_root=str(tmp_path), name="rts")
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    storage.append_summary(camp, {"round": 0, "eval_speed": 0.25, "eval_cost_kevals": 12.0})
    df = storage.read_summary(camp)
    assert "eval_speed" in df.columns
    assert float(df.iloc[0]["eval_speed"]) == pytest.approx(0.25)
