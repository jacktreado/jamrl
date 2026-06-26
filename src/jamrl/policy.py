"""Policy/value nets + observation normalizer + (de)serialization (plan 5.4).

Backend note: the plan specifies PyTorch, but the nets are tiny 2-hidden-layer
MLPs, so this uses a self-contained NumPy implementation (manual backprop +
Adam). This keeps the learner runnable where pip-torch conflicts with the env's
OpenMP/MKL. The actor-facing weights are written to npz exactly as the C++
``_core.Policy`` expects, so the engine is unaffected by the backend choice.
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 10  # must match jamcore::OBS_DIM
ACT_DIM = 2   # must match jamcore::ACT_DIM


# ----------------------------------------------------------------------- #
class RunningNorm:
    """Welford running mean/var for observation normalization."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return
        bn = X.shape[0]
        bmean = X.mean(axis=0)
        bvar = X.var(axis=0)
        delta = bmean - self.mean
        tot = self.count + bn
        self.mean += delta * bn / tot
        m_a = self.var * self.count
        m_b = bvar * bn
        self.var = (m_a + m_b + delta**2 * self.count * bn / tot) / tot
        self.count = tot

    def normalize(self, X):
        return (np.asarray(X) - self.mean) / np.sqrt(self.var + 1e-8)

    def state(self):
        return dict(mean=self.mean.copy(), var=self.var.copy(), count=np.float64(self.count))

    def load(self, st):
        self.mean = np.asarray(st["mean"], float).copy()
        self.var = np.asarray(st["var"], float).copy()
        self.count = float(st["count"])


# ----------------------------------------------------------------------- #
class MLP:
    """Fully-connected net, tanh hidden activations, linear output."""

    def __init__(self, sizes, seed=0, scale=None):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            s = scale if scale is not None else np.sqrt(1.0 / sizes[i])
            self.W.append(rng.standard_normal((sizes[i + 1], sizes[i])) * s)
            self.b.append(np.zeros(sizes[i + 1]))
        self.n = len(self.W)

    def forward(self, X):
        self.z, self.a = [], [np.asarray(X, dtype=np.float64)]
        a = self.a[0]
        for i in range(self.n):
            z = a @ self.W[i].T + self.b[i]
            self.z.append(z)
            a = np.tanh(z) if i < self.n - 1 else z
            self.a.append(a)
        return a

    def backward(self, dout):
        """dout = dLoss/d(output) [B, out]; returns (dW list, db list)."""
        dW = [None] * self.n
        db = [None] * self.n
        delta = dout
        for i in reversed(range(self.n)):
            a_prev = self.a[i]
            dW[i] = delta.T @ a_prev
            db[i] = delta.sum(axis=0)
            if i > 0:
                da = delta @ self.W[i]
                delta = da * (1.0 - np.tanh(self.z[i - 1]) ** 2)
        return dW, db

    def params(self):
        return [*self.W, *self.b]

    def grads_to_list(self, dW, db):
        return [*dW, *db]


# ----------------------------------------------------------------------- #
class PolicyNet:
    def __init__(self, obs_dim=OBS_DIM, hidden=(64, 64), act_dim=ACT_DIM,
                 logstd_init=-0.5, seed=0):
        assert len(hidden) == 2, "C++ runner supports exactly 2 hidden layers"
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = tuple(hidden)
        self.mlp = MLP([obs_dim, hidden[0], hidden[1], act_dim], seed=seed, scale=0.1)
        self.log_std = np.full(act_dim, float(logstd_init))

    def mu(self, Xn):
        return self.mlp.forward(Xn)

    def parameters(self):
        return [*self.mlp.params(), self.log_std]


class ValueNet:
    def __init__(self, obs_dim=OBS_DIM, hidden=(64, 64), seed=1):
        self.mlp = MLP([obs_dim, hidden[0], hidden[1], 1], seed=seed, scale=0.1)

    def value(self, Xn):
        return self.mlp.forward(Xn)[:, 0]

    def parameters(self):
        return self.mlp.params()


# ----------------------------------------------------------------------- #
class Adam:
    def __init__(self, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.t = 0
        self.m, self.v = {}, {}

    def step(self, params, grads, max_norm=None):
        self.t += 1
        if max_norm is not None:
            gn = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
            scale = min(1.0, max_norm / (gn + 1e-12))
        else:
            scale = 1.0
        for p, g in zip(params, grads):
            g = g * scale
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g * g)
            mhat = self.m[key] / (1 - self.b1 ** self.t)
            vhat = self.v[key] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state(self):
        return {"t": self.t}  # moments are re-warmed on resume (cheap, tiny nets)

    def load(self, st):
        self.t = int(st.get("t", 0))


# ----------------------------------------------------------------------- #
# Serialization
# ----------------------------------------------------------------------- #
def policy_arrays(policy: PolicyNet, norm: RunningNorm) -> dict:
    W0, W1, Wmu = policy.mlp.W
    b0, b1, bmu = policy.mlp.b
    return {
        "obs_mean": norm.mean.astype(np.float64),
        "obs_std": np.sqrt(norm.var + 1e-8).astype(np.float64),
        "W0": W0.astype(np.float64), "b0": b0.astype(np.float64),
        "W1": W1.astype(np.float64), "b1": b1.astype(np.float64),
        "Wmu": Wmu.astype(np.float64), "bmu": bmu.astype(np.float64),
        "log_std": policy.log_std.astype(np.float64),
    }


def save_policy_npz(path, policy: PolicyNet, norm: RunningNorm):
    from jamrl.storage import atomic_path

    arrs = policy_arrays(policy, norm)
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.savez(f, **arrs)


def load_policy_npz(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_core_policy(d: dict):
    """Build a _core.Policy from a loaded actor npz dict."""
    from jamrl import _core

    return _core.Policy(d["obs_mean"], d["obs_std"], d["W0"], d["b0"], d["W1"], d["b1"],
                        d["Wmu"], d["bmu"], d["log_std"])


def init_policy_npz(path, obs_dim=OBS_DIM, hidden=(64, 64), act_dim=ACT_DIM,
                    logstd_init=-0.5, seed=0):
    """Write a fresh random initial policy (round 0)."""
    pol = PolicyNet(obs_dim, hidden, act_dim, logstd_init, seed=seed)
    norm = RunningNorm(obs_dim)
    save_policy_npz(path, pol, norm)
    return pol, norm
