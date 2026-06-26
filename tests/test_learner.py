"""Phase 8 gate: PPO/CEM learning, checkpoint round-trip, run-local (plan 8)."""
import os

import numpy as np
import pytest

from jamrl import cem as cem_mod
from jamrl import cli, learn, policy, ppo, rollout, storage
from jamrl.config import Config
from jamrl.policy import Adam, PolicyNet, RunningNorm, ValueNet


# --------------------------------------------------------------------------- #
def test_ppo_optimizes_bandit():
    """PPO drives a Gaussian policy toward the optimum on a contextual bandit."""
    cfg = Config(gamma=0.0, lam=0.0, clip=0.2, ppo_epochs=10, minibatch=256,
                 ent_coef=0.0, vf_coef=0.5, lr=3e-3, seed=0)
    pol = PolicyNet(obs_dim=4, hidden=(32, 32), act_dim=2, logstd_init=-0.5, seed=0)
    val = ValueNet(obs_dim=4, hidden=(32, 32), seed=1)
    norm = RunningNorm(4)
    op, ov = Adam(cfg.lr), Adam(cfg.lr)
    rng = np.random.default_rng(0)

    def target(o):
        return np.stack([np.tanh(o[:, 0]), -np.tanh(o[:, 1])], axis=1)

    def greedy(O):
        mg = np.clip(pol.mu(norm.normalize(O)), -1, 1)
        return -np.mean(np.sum((mg - target(O)) ** 2, axis=1))

    O0 = rng.standard_normal((256, 4))
    r_start = greedy(O0)
    for _ in range(40):
        O = rng.standard_normal((256, 4))
        mu = pol.mu(norm.normalize(O))
        A = mu + np.exp(pol.log_std) * rng.standard_normal(mu.shape)
        R = -np.sum((np.clip(A, -1, 1) - target(O)) ** 2, axis=1)
        traj = dict(obs=O.astype(np.float32), act=A.astype(np.float32),
                    rew=R.astype(np.float32), done=np.ones(256, bool),
                    ep_ptr=np.arange(257, dtype=np.int64))
        ppo.ppo_update(cfg, pol, val, norm, op, ov, traj)
    assert greedy(O0) > r_start + 0.2


def test_cem_elite_nondecreasing():
    """CEM elite mean improves and converges to the optimum on a quadratic."""
    target = np.array([1.0, -2.0, 0.5])
    cem = cem_mod.CEM(3, np.zeros(3), sigma0=1.5, elite_frac=0.25)
    rng = np.random.default_rng(0)
    elites = []
    for _ in range(40):
        c = cem.ask(64, rng)
        scores = -np.sum((c - target) ** 2, axis=1)
        elites.append(cem.tell(c, scores))
    assert elites[-1] > elites[0]
    # overall trend strongly increasing (allow minor sampling dips)
    assert np.mean(elites[-3:]) > np.mean(elites[:3])
    assert np.allclose(cem.mu, target, atol=0.3)


@pytest.mark.slow
def test_ppo_jamming_reward_improves(tmp_path):
    """PPO improves mean episode reward on the N=32 jamming task."""
    cfg = Config(N=32, P=1e-3, T_cap=15, n_relax=60, workers=4, episodes_per_worker=8,
                 rounds=20, save_hessian="none", hidden=(32, 32), minibatch=512,
                 ppo_epochs=6, lr=1.5e-3, ent_coef=1e-3, logstd_init=0.0, backend="numpy",
                 campaign_root=str(tmp_path), name="ppo_jam", seed=1)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden,
                           logstd_init=cfg.logstd_init, seed=cfg.seed)
    rewards = []
    for r in range(cfg.rounds):
        for k in range(cfg.workers):
            rollout.run_rollout(cfg, camp, r, k)
        rewards.append(learn.learn_round(cfg, camp, r)["mean_reward"])
    assert np.mean(rewards[-5:]) > np.mean(rewards[:5]) + 0.3, rewards


def test_checkpoint_roundtrip(tmp_path):
    cfg = Config(N=32, T_cap=10, n_relax=20, workers=2, episodes_per_worker=2, rounds=1,
                 save_hessian="none", hidden=(16, 16), backend="numpy",
                 campaign_root=str(tmp_path), name="ck", seed=5)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)
    for k in range(cfg.workers):
        rollout.run_rollout(cfg, camp, 0, k)
    learn.learn_round(cfg, camp, 0)  # writes policy_1 + checkpoint_1

    pol = PolicyNet(hidden=(16, 16))
    val = ValueNet(hidden=(16, 16))
    norm = RunningNorm(10)
    learn.load_checkpoint(storage.checkpoint_path(camp, 1), pol, val, norm, Adam(), Adam())
    d = policy.load_policy_npz(storage.policy_path(camp, 1))
    core = policy.build_core_policy(d)
    rng = np.random.default_rng(2)
    for _ in range(5):
        o = rng.standard_normal(10)
        assert np.allclose(pol.mu(norm.normalize(o[None]))[0], core.forward(o), atol=1e-10)


def test_resume_matches_uninterrupted(tmp_path):
    def run(name):
        cfg = Config(N=32, T_cap=10, n_relax=20, workers=2, episodes_per_worker=2, rounds=2,
                     save_hessian="none", hidden=(16, 16), backend="numpy",
                     campaign_root=str(tmp_path), name=name, seed=5)
        camp = storage.campaign_dir(cfg)
        storage.ensure_campaign_dirs(camp)
        cfg.save_yaml(os.path.join(camp, "config.yaml"))
        policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)
        for r in range(2):
            for k in range(cfg.workers):
                rollout.run_rollout(cfg, camp, r, k)
            learn.learn_round(cfg, camp, r)
        return policy.load_policy_npz(storage.policy_path(camp, 2))

    a, b = run("A"), run("B")
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_run_local_end_to_end(tmp_path):
    """`jamrl run-local` completes 2 rounds at N=64 and writes valid data."""
    rc = cli.main([
        "run-local", "--N", "64", "--rounds", "2", "--workers", "2",
        "--episodes-per-worker", "2", "--T-cap", "12", "--n-relax", "20",
        "--hidden", "16,16", "--save-hessian", "none",
        "--campaign-root", str(tmp_path), "--name", "smoke",
    ])
    assert rc == 0
    camp = os.path.join(str(tmp_path), "smoke")
    assert os.path.exists(storage.policy_path(camp, 0))
    assert os.path.exists(storage.policy_path(camp, 2))  # produced through round 1
    assert os.path.exists(os.path.join(camp, "DONE"))
    df = storage.read_summary(camp)
    assert len(df) == 2
    assert np.isfinite(df["mean_reward"]).all()
    assert np.isfinite(df["eval_success"]).all()


# --------------------------------------------------------------------------- #
# Torch backend (plan-specified PyTorch path)
# --------------------------------------------------------------------------- #
from jamrl import torch_backend as tb  # noqa: E402

torch_only = pytest.mark.skipif(not tb.HAS_TORCH, reason="torch unavailable")


@torch_only
def test_torch_ppo_optimizes_bandit():
    import torch

    cfg = Config(gamma=0.0, lam=0.0, clip=0.2, ppo_epochs=10, minibatch=256,
                 ent_coef=0.0, vf_coef=0.5, lr=3e-3, seed=0)
    pol = tb.TorchPolicy(obs_dim=4, hidden=(32, 32), act_dim=2, logstd_init=-0.5)
    val = tb.TorchValue(obs_dim=4, hidden=(32, 32))
    norm = RunningNorm(4)
    op = torch.optim.Adam(pol.parameters(), lr=cfg.lr)
    ov = torch.optim.Adam(val.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(0)

    def target(o):
        return np.stack([np.tanh(o[:, 0]), -np.tanh(o[:, 1])], axis=1)

    def greedy(O):
        with torch.no_grad():
            mg = pol(torch.as_tensor(norm.normalize(O).astype(np.float32))).numpy()
        return -np.mean(np.sum((np.clip(mg, -1, 1) - target(O)) ** 2, axis=1))

    O0 = rng.standard_normal((256, 4))
    r0 = greedy(O0)
    for _ in range(40):
        O = rng.standard_normal((256, 4))
        with torch.no_grad():
            mu = pol(torch.as_tensor(norm.normalize(O).astype(np.float32))).numpy()
        A = mu + np.exp(pol.log_std.detach().numpy()) * rng.standard_normal(mu.shape)
        R = -np.sum((np.clip(A, -1, 1) - target(O)) ** 2, axis=1)
        traj = dict(obs=O.astype(np.float32), act=A.astype(np.float32),
                    rew=R.astype(np.float32), done=np.ones(256, bool),
                    ep_ptr=np.arange(257, dtype=np.int64))
        tb.ppo_update(cfg, pol, val, norm, op, ov, traj)
    assert greedy(O0) > r0 + 0.2


@torch_only
def test_torch_policy_npz_matches_core():
    """Torch policy exported to npz reproduces in the C++ runner (bit-close)."""
    import torch

    pol = tb.TorchPolicy(hidden=(16, 16), logstd_init=-0.5)
    norm = RunningNorm(10)
    rng = np.random.default_rng(3)
    norm.update(rng.standard_normal((64, 10)))  # nontrivial normalizer
    d = tb.export_npz(pol, norm)
    core = policy.build_core_policy(d)
    for _ in range(5):
        o = rng.standard_normal(10)
        with torch.no_grad():
            mu_t = pol(torch.as_tensor(norm.normalize(o[None]).astype(np.float32))).numpy()[0]
        assert np.allclose(core.forward(o), mu_t, atol=1e-6)


@torch_only
def test_torch_checkpoint_roundtrip(tmp_path):
    import torch

    cfg = Config(N=32, T_cap=10, n_relax=20, workers=2, episodes_per_worker=2, rounds=1,
                 save_hessian="none", hidden=(16, 16), backend="torch",
                 campaign_root=str(tmp_path), name="ckt", seed=5)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)
    for k in range(cfg.workers):
        rollout.run_rollout(cfg, camp, 0, k)
    learn.learn_round(cfg, camp, 0)
    assert os.path.exists(tb.checkpoint_path(camp, 1))  # .pt checkpoint written

    pol = tb.TorchPolicy(hidden=(16, 16))
    val = tb.TorchValue(hidden=(16, 16))
    norm = RunningNorm(10)
    op = torch.optim.Adam(pol.parameters())
    ov = torch.optim.Adam(val.parameters())
    tb.load_checkpoint(tb.checkpoint_path(camp, 1), pol, val, norm, op, ov)
    core = policy.build_core_policy(policy.load_policy_npz(storage.policy_path(camp, 1)))
    rng = np.random.default_rng(2)
    for _ in range(5):
        o = rng.standard_normal(10)
        with torch.no_grad():
            mu_t = pol(torch.as_tensor(norm.normalize(o[None]).astype(np.float32))).numpy()[0]
        assert np.allclose(core.forward(o), mu_t, atol=1e-6)


@torch_only
@pytest.mark.slow
def test_torch_jamming_reward_improves(tmp_path):
    cfg = Config(N=32, P=1e-3, T_cap=15, n_relax=60, workers=4, episodes_per_worker=8,
                 rounds=20, save_hessian="none", hidden=(32, 32), minibatch=512,
                 ppo_epochs=6, lr=1.5e-3, ent_coef=1e-3, logstd_init=0.0, backend="torch",
                 campaign_root=str(tmp_path), name="ppo_jam_t", seed=1)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden,
                           logstd_init=cfg.logstd_init, seed=cfg.seed)
    rewards = []
    for r in range(cfg.rounds):
        for k in range(cfg.workers):
            rollout.run_rollout(cfg, camp, r, k)
        rewards.append(learn.learn_round(cfg, camp, r)["mean_reward"])
    assert np.mean(rewards[-5:]) > np.mean(rewards[:5]) + 0.3, rewards

