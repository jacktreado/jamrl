"""CEM learner: population search over flattened policy params (plan 5.7).

No value net, no gradients. Each candidate is a full flattened PolicyNet
parameter vector sampled from N(mu, diag(sigma^2)); candidates are scored on a
shared seed set (low-variance comparison); mu/sigma refit to the elites; the
distribution mean ships as the next round's policy.
"""
from __future__ import annotations

import numpy as np

from jamrl import config as cfg_mod
from jamrl.policy import PolicyNet, RunningNorm


def flatten_policy(pol: PolicyNet) -> np.ndarray:
    parts = [pol.mlp.W[0].ravel(), pol.mlp.b[0], pol.mlp.W[1].ravel(), pol.mlp.b[1],
             pol.mlp.W[2].ravel(), pol.mlp.b[2], pol.log_std]
    return np.concatenate([np.asarray(p, float).ravel() for p in parts])


def unflatten_into(pol: PolicyNet, flat: np.ndarray) -> PolicyNet:
    i = 0
    shapes = [pol.mlp.W[0].shape, pol.mlp.b[0].shape, pol.mlp.W[1].shape,
              pol.mlp.b[1].shape, pol.mlp.W[2].shape, pol.mlp.b[2].shape, pol.log_std.shape]
    out = []
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(flat[i:i + n].reshape(sh).copy())
        i += n
    pol.mlp.W[0], pol.mlp.b[0], pol.mlp.W[1], pol.mlp.b[1], pol.mlp.W[2], pol.mlp.b[2], pol.log_std = out
    return pol


class CEM:
    def __init__(self, dim, mu, sigma0, elite_frac, sigma_floor=1e-3):
        self.dim = dim
        self.mu = np.asarray(mu, float).copy()
        self.sigma = np.full(dim, float(sigma0))
        self.elite_frac = elite_frac
        self.sigma_floor = sigma_floor

    def ask(self, pop, rng) -> np.ndarray:
        return self.mu[None, :] + self.sigma[None, :] * rng.standard_normal((pop, self.dim))

    def tell(self, candidates, scores):
        scores = np.asarray(scores, float)
        n_elite = max(1, int(round(self.elite_frac * len(scores))))
        elite_idx = np.argsort(scores)[::-1][:n_elite]
        elites = candidates[elite_idx]
        self.mu = elites.mean(axis=0)
        self.sigma = np.maximum(elites.std(axis=0), self.sigma_floor)
        return float(scores[elite_idx].mean())

    def state(self):
        return {"mu": self.mu.copy(), "sigma": self.sigma.copy()}

    def load(self, st):
        self.mu = np.asarray(st["mu"], float).copy()
        self.sigma = np.asarray(st["sigma"], float).copy()


def make_template(cfg) -> PolicyNet:
    return PolicyNet(hidden=cfg.hidden, act_dim=cfg_mod.act_dim(cfg),
                     logstd_init=cfg.logstd_init, seed=cfg.seed)


def candidate_policy(template: PolicyNet, flat) -> PolicyNet:
    pol = PolicyNet(hidden=template.hidden, act_dim=template.act_dim, logstd_init=0.0, seed=0)
    return unflatten_into(pol, flat)
