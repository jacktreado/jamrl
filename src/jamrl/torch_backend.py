"""PyTorch learner backend (plan section 5.4/5.6).

The default PPO backend. Mirrors the NumPy implementation (same GAE / clipped
surrogate / value loss / entropy) but uses autograd. The actor-facing policy is
exported to the *same* npz the C++ ``_core.Policy`` consumes, so the engine and
the CEM path are unaffected by the choice of backend. Checkpoints are ``.pt``.

Importing this module does not require torch; ``HAS_TORCH`` is False if it is
unavailable, and learn.py falls back to the NumPy backend.
"""
from __future__ import annotations

import os

import numpy as np

from jamrl import config as cfg_mod
from jamrl import policy as pol_mod
from jamrl import ppo, storage
from jamrl.policy import OBS_DIM, RunningNorm

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None
    HAS_TORCH = False

LOG2PI = float(np.log(2.0 * np.pi))


def resolve_backend(cfg) -> str:
    want = getattr(cfg, "backend", "auto")
    if want == "numpy":
        return "numpy"
    if want == "torch":
        if not HAS_TORCH:
            raise RuntimeError("backend='torch' requested but torch is unavailable")
        return "torch"
    return "torch" if HAS_TORCH else "numpy"  # auto


if HAS_TORCH:

    class TorchPolicy(nn.Module):
        def __init__(self, obs_dim=OBS_DIM, hidden=(64, 64), act_dim=2, logstd_init=-0.5):
            super().__init__()
            assert len(hidden) == 2, "C++ runner supports exactly 2 hidden layers"
            self.l0 = nn.Linear(obs_dim, hidden[0])
            self.l1 = nn.Linear(hidden[0], hidden[1])
            self.lmu = nn.Linear(hidden[1], act_dim)
            self.log_std = nn.Parameter(torch.full((act_dim,), float(logstd_init)))
            for layer in (self.l0, self.l1, self.lmu):
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)

        def forward(self, xn):
            h = torch.tanh(self.l0(xn))
            h = torch.tanh(self.l1(h))
            return self.lmu(h)

    class TorchValue(nn.Module):
        def __init__(self, obs_dim=OBS_DIM, hidden=(64, 64)):
            super().__init__()
            self.l0 = nn.Linear(obs_dim, hidden[0])
            self.l1 = nn.Linear(hidden[0], hidden[1])
            self.lo = nn.Linear(hidden[1], 1)
            for layer in (self.l0, self.l1, self.lo):
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)

        def forward(self, xn):
            h = torch.tanh(self.l0(xn))
            h = torch.tanh(self.l1(h))
            return self.lo(h).squeeze(-1)

    def _gauss_logp(act, mu, log_std):
        std = torch.exp(log_std)
        z = (act - mu) / std
        return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * act.shape[1] * LOG2PI

    def export_npz(policy: "TorchPolicy", norm: RunningNorm) -> dict:
        def npy(t):
            return t.detach().cpu().numpy().astype(np.float64)

        return {
            "obs_mean": norm.mean.astype(np.float64),
            "obs_std": np.sqrt(norm.var + 1e-8).astype(np.float64),
            "W0": npy(policy.l0.weight), "b0": npy(policy.l0.bias),
            "W1": npy(policy.l1.weight), "b1": npy(policy.l1.bias),
            "Wmu": npy(policy.lmu.weight), "bmu": npy(policy.lmu.bias),
            "log_std": npy(policy.log_std),
        }

    def save_policy_npz(path, policy, norm):
        with storage.atomic_path(path) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, **export_npz(policy, norm))

    def _load_policy_from_npz(policy, d):
        with torch.no_grad():
            policy.l0.weight.copy_(torch.as_tensor(d["W0"], dtype=torch.float32))
            policy.l0.bias.copy_(torch.as_tensor(d["b0"], dtype=torch.float32))
            policy.l1.weight.copy_(torch.as_tensor(d["W1"], dtype=torch.float32))
            policy.l1.bias.copy_(torch.as_tensor(d["b1"], dtype=torch.float32))
            policy.lmu.weight.copy_(torch.as_tensor(d["Wmu"], dtype=torch.float32))
            policy.lmu.bias.copy_(torch.as_tensor(d["bmu"], dtype=torch.float32))
            policy.log_std.copy_(torch.as_tensor(d["log_std"], dtype=torch.float32))

    def ppo_update(cfg, policy, value, norm, opt_p, opt_v, traj) -> dict:
        obs = np.asarray(traj["obs"], np.float64)
        act = np.asarray(traj["act"], np.float64)
        rew = np.asarray(traj["rew"], np.float64)
        done = np.asarray(traj["done"], bool)
        ep_ptr = np.asarray(traj["ep_ptr"], np.int64)
        T = obs.shape[0]
        if T == 0:
            return {"mean_reward": 0.0, "n_transitions": 0}

        norm.update(obs)
        Xn = norm.normalize(obs).astype(np.float32)
        Xt = torch.as_tensor(Xn)
        at = torch.as_tensor(act.astype(np.float32))

        with torch.no_grad():
            values = value(Xt).cpu().numpy().astype(np.float64)
        adv, returns = ppo.compute_gae(rew, values, done, ep_ptr, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv.astype(np.float32))
        ret_t = torch.as_tensor(returns.astype(np.float32))
        with torch.no_grad():
            logp_old = _gauss_logp(at, policy(Xt), policy.log_std)

        mb = max(1, min(cfg.minibatch, T))
        gen = torch.Generator().manual_seed(int(cfg.seed) + 1000)
        last_kl = 0.0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(T, generator=gen)
            for s in range(0, T, mb):
                idx = perm[s:s + mb]
                mu = policy(Xt[idx])
                logp = _gauss_logp(at[idx], mu, policy.log_std)
                ratio = torch.exp(logp - logp_old[idx])
                a = adv_t[idx]
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a)
                ent = (policy.log_std + 0.5 * (1.0 + LOG2PI)).sum()
                loss_pi = -surr.mean() - cfg.ent_coef * ent
                opt_p.zero_grad(set_to_none=True)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt_p.step()
                with torch.no_grad():  # exploration floor/ceiling on log_std
                    policy.log_std.clamp_(min=cfg.logstd_min, max=cfg.logstd_max)

                vpred = value(Xt[idx])
                loss_v = cfg.vf_coef * 0.5 * ((vpred - ret_t[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                loss_v.backward()
                nn.utils.clip_grad_norm_(value.parameters(), 1.0)
                opt_v.step()
            with torch.no_grad():
                last_kl = float((logp_old - _gauss_logp(at, policy(Xt), policy.log_std)).mean())

        ep_returns = [float(rew[ep_ptr[e]:ep_ptr[e + 1]].sum()) for e in range(len(ep_ptr) - 1)]
        return {
            "mean_reward": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_transitions": int(T),
            "approx_kl": last_kl,
            "sigma_policy": float(torch.exp(policy.log_std).mean().item()),
        }

    def checkpoint_path(camp, r):
        return os.path.join(camp, "checkpoints", f"learner_round_{r:04d}.pt")

    def save_checkpoint(path, policy, value, norm, opt_p, opt_v, rnd):
        blob = {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "opt_p": opt_p.state_dict(),
            "opt_v": opt_v.state_dict(),
            "norm": norm.state(),
            "round": rnd,
        }
        with storage.atomic_path(path) as tmp:
            torch.save(blob, tmp)

    def load_checkpoint(path, policy, value, norm, opt_p, opt_v):
        blob = torch.load(path, weights_only=False)
        policy.load_state_dict(blob["policy"])
        value.load_state_dict(blob["value"])
        opt_p.load_state_dict(blob["opt_p"])
        opt_v.load_state_dict(blob["opt_v"])
        norm.load(blob["norm"])

    def learn_ppo_round(cfg, camp, r, traj) -> dict:
        """Torch PPO for one round: load/init -> update -> write npz + .pt."""
        policy = TorchPolicy(hidden=cfg.hidden, act_dim=cfg_mod.act_dim(cfg),
                             logstd_init=cfg.logstd_init)
        value = TorchValue(hidden=cfg.hidden)
        norm = RunningNorm(OBS_DIM)
        opt_p = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        opt_v = torch.optim.Adam(value.parameters(), lr=cfg.lr)

        ckpt = checkpoint_path(camp, r)
        if os.path.exists(ckpt):
            load_checkpoint(ckpt, policy, value, norm, opt_p, opt_v)
        else:
            d = pol_mod.load_policy_npz(storage.policy_path(camp, r))
            _load_policy_from_npz(policy, d)
            norm.mean = np.asarray(d["obs_mean"], float).copy()
            norm.var = np.maximum(np.asarray(d["obs_std"], float) ** 2 - 1e-8, 1e-8)

        stats = ppo_update(cfg, policy, value, norm, opt_p, opt_v, traj)
        save_policy_npz(storage.policy_path(camp, r + 1), policy, norm)
        save_checkpoint(checkpoint_path(camp, r + 1), policy, value, norm, opt_p, opt_v, r + 1)
        return stats
