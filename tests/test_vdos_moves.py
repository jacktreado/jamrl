"""Gate: VDOS-directed agent moves (plan (c)).

The agent emits `k_vdos_moves` extra action coefficients that displace particles
along the lowest soft modes each macro-step (x += vdos_move_amp * c_j * e_j),
before the held-load relaxation. 0 disables (ACT_DIM stays 2).
"""
import argparse
import os
import tempfile

import numpy as np

import jamrl._core as core
from jamrl import config, policy
from jamrl.config import Config


def _widened_policy(cfg, seed=0):
    """A fresh policy whose action width is 2 + k_vdos_moves."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p0.npz")
        policy.init_policy_npz(p, hidden=(16, 16), act_dim=config.act_dim(cfg),
                               logstd_init=-0.5, seed=seed)
        return policy.build_core_policy(policy.load_policy_npz(p))


def _run(cfg, seeds):
    ec = config.env_config(cfg)
    pol = _widened_policy(cfg)
    proto = core.make_system(cfg.N, seeds[0], cfg.phi0, cfg.P)
    sf = core.SaveFlags(); sf.save_hessian = 0; sf.save_moduli = True; sf.save_contacts = False
    return core.run_episodes_batch(proto, pol, seeds, ec, sf, 1,
                                   [1.0] * len(seeds), [1.0] * len(seeds))


# --------------------------------------------------------------------------- #
# 1. Config plumbing: CLI flags, env_config bridge, act_dim helper.
# --------------------------------------------------------------------------- #
def test_vdos_moves_config_plumbing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    config.add_arguments(parser)
    cfg = config.from_args(parser.parse_args(["--k-vdos-moves", "5", "--vdos-move-amp", "0.03"]))
    assert cfg.k_vdos_moves == 5
    assert cfg.vdos_move_amp == 0.03
    assert config.act_dim(cfg) == 7            # 2 box + 5 moves
    assert config.act_dim(Config()) == 2       # default disables the feature

    ec = config.env_config(cfg)
    assert ec.k_vdos_moves == 5
    assert ec.vdos_move_amp == 0.03


# --------------------------------------------------------------------------- #
# 2. Action width follows k_vdos_moves end-to-end (policy -> C++ -> stored act).
# --------------------------------------------------------------------------- #
def test_vdos_moves_widen_action():
    seeds = list(range(11, 23))
    on = Config(N=32, P=1e-3, phi0=0.80, reward_mode="shear_modulus",
                k_vdos_moves=5, vdos_move_amp=0.05, vdos_obs=True)
    off = Config(N=32, P=1e-3, phi0=0.80, reward_mode="shear_modulus")  # default: 0 moves
    assert _run(on, seeds)[0]["act"].shape[1] == 7
    assert _run(off, seeds)[0]["act"].shape[1] == 2


# --------------------------------------------------------------------------- #
# 3. The kick is wired to the physics: enabling it (amp>0) perturbs the
#    dynamics vs the same policy/seeds with amp=0 (kick scaled to zero).
# --------------------------------------------------------------------------- #
def test_vdos_kick_changes_dynamics():
    seeds = list(range(11, 43))
    base = dict(N=64, P=1e-3, phi0=0.80, reward_mode="shear_modulus",
                k_vdos_moves=5, vdos_obs=True)
    eps_off = _run(Config(**base, vdos_move_amp=0.0), seeds)   # kick disabled
    eps_on = _run(Config(**base, vdos_move_amp=0.15), seeds)   # kick enabled

    assert eps_off[0]["act"].shape[1] == 7 and eps_on[0]["act"].shape[1] == 7
    diverged = sum(
        1 for a, b in zip(eps_off, eps_on)
        if a["act"].shape[0] != b["act"].shape[0] or not np.allclose(a["act"], b["act"])
    )
    # amp scales the displacement, so amp=0 is a true no-op control; a nonzero kick
    # must move the trajectory for the great majority of seeds.
    assert diverged >= len(seeds) // 2, f"only {diverged}/{len(seeds)} trajectories diverged"
