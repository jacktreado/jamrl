"""Phase 3 gate: the MDP environment (plan section 3.5, 4.2-4.4, 8)."""
import numpy as np
import pytest

import jamrl._core as core

OC = dict(core.OUTCOMES)


def make_env(cfg=None):
    e = core.Env()
    if cfg is not None:
        e.cfg = cfg
    return e


def jam(seed, N=32, P=1e-3):
    s = core.make_system(N, seed, 0.80, P)
    core.relax(s, 0.0, 0.0, 20000)
    return s


# --------------------------------------------------------------------------- #
def test_null_episodes_jam_to_physical_density():
    """Zero-action episodes jam reproducibly to a 2D bidisperse density.

    Prototype regression anchors at N=32, P=1e-3 are ~0.845/0.824/0.841 for
    seeds 1/2/101; the exact values depend on the prototype's init RNG (not
    reproduced here), so we gate on the physical window + determinism.
    """
    cfg = core.EnvConfig()
    vals = {}
    for seed in (1, 2, 101):
        proto = core.make_system(32, seed, 0.80, 1e-3)
        phin = core.compute_null_phi(proto, cfg)
        vals[seed] = phin
        assert 0.80 < phin < 0.88, f"seed {seed}: phi_null={phin}"
        # determinism
        assert core.compute_null_phi(core.make_system(32, seed, 0.80, 1e-3), cfg) == phin


def test_null_episode_reward_near_baseline():
    """A real null episode (phi == phi_null) earns only the small step/trunc cost."""
    cfg = core.EnvConfig()
    proto = core.make_system(32, 1, 0.80, 1e-3)
    phin = core.compute_null_phi(proto, cfg)
    env = make_env(cfg)
    env.reset(proto, phin)
    while not env.done:
        tr = env.step(0.0, 0.0)
    assert OC["quiesced"] == env.outcome or OC["converged"] == env.outcome
    # density term ~0; only -trunc_pen - c_step*steps remains
    assert -1.0 < env.total_reward < 0.0


def test_action_semantics_pressure():
    """Held a_P=+1 drives (P_int - P)/P toward kappa_P."""
    cfg = core.EnvConfig()
    proto = jam(1)
    env = make_env(cfg)
    env.reset(proto, 0.83)
    last = 0.0
    for _ in range(25):
        tr = env.step(1.0, 0.0)
        last = (tr["P_int"] - 1e-3) / 1e-3
        if tr["done"]:
            break
    assert last > 0.5, f"(P_int-P)/P={last} should approach kappa_P=1"
    assert last < 1.6


def test_action_semantics_shear():
    """Held a_sigma drives shear stress in the forcing direction.

    gamma itself wraps into (-1/2, 1/2], so the wrap-invariant signature of the
    shear input is the sign of the shear-stress residual Egamma -> sigF.
    """
    cfg = core.EnvConfig()

    def run_shear(asig):
        proto = jam(2)
        env = make_env(cfg)
        env.reset(proto, 0.83)
        eg = 0.0
        for _ in range(3):
            tr = env.step(0.0, asig)
            eg = tr["Egamma"]
            if tr["done"]:
                break
        return eg

    eg_pos = run_shear(+1.0)
    eg_neg = run_shear(-1.0)
    assert eg_pos > 1e-4, f"Egamma under +shear={eg_pos}"
    assert eg_neg < -1e-4, f"Egamma under -shear={eg_neg}"


def test_pressure_floor_no_box_runaway():
    """Fix 4.2: full decompression (a_P=-1) keeps L bounded, no blowup."""
    cfg = core.EnvConfig()
    proto = jam(3)
    env = make_env(cfg)
    env.reset(proto, 0.83)
    Lmax = proto.L
    while not env.done:
        tr = env.step(-1.0, 0.0)
        Lmax = max(Lmax, env.sys.L)
    assert Lmax < 1e3, f"L blew up to {Lmax} (pressure floor failed)"
    assert env.outcome not in (OC["blowup"],)


def test_melt_detection_on_dilated_state():
    """Fix 4.2: a state that stays at phi < 0.3 after a macro-step ends as melt.

    With the default n_relax=20 the minimizer recompresses a dilated box within
    one step, so melt is a rare safeguard; we exercise the classification branch
    with a short relax budget where recompression cannot keep up.
    """
    cfg = core.EnvConfig()
    cfg.n_relax = 1
    proto = core.make_system(32, 7, 0.80, 1e-3)
    x = np.asarray(proto.x)
    x[2 * 32] = np.log(proto.L * 6.0)  # inflate box ~36x area -> phi ~ 0.022
    proto.x = x
    assert proto.phi < 0.3
    env = make_env(cfg)
    env.reset(proto, 0.83)
    tr = env.step(0.0, 0.0)
    assert tr["done"]
    assert env.outcome == OC["melt"], f"outcome={env.outcome}"


def test_finish_and_measure_converged():
    """Releasing at a jammed state yields outcome=converged."""
    cfg = core.EnvConfig()
    proto = jam(11)
    env = make_env(cfg)
    env.reset(proto, proto.phi)
    tr = env.step(0.0, 0.0)
    assert tr["done"]
    assert env.outcome == OC["converged"], f"outcome={env.outcome}"


def test_episode_determinism_bitwise():
    """Same proto + same action sequence -> identical transitions, bit for bit."""
    cfg = core.EnvConfig()
    acts = [(0.4, -0.2), (0.1, 0.3), (-0.5, 0.5), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

    def run():
        proto = core.make_system(32, 9, 0.80, 1e-3)
        env = make_env(cfg)
        env.reset(proto, 0.83)
        obs_log, rew_log = [], []
        for a in acts:
            tr = env.step(*a)
            obs_log.append(np.asarray(tr["obs"]))
            rew_log.append(tr["reward"])
            if tr["done"]:
                break
        return np.array(obs_log), np.array(rew_log)

    o1, r1 = run()
    o2, r2 = run()
    assert np.array_equal(o1, o2)
    assert np.array_equal(r1, r2)
