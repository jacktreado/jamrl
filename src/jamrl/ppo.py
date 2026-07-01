"""PPO learner: GAE(lambda) + clipped surrogate + value loss + entropy (plan 5.6).

NumPy backend (see policy.py). The actor stores raw pre-clip actions; the
Gaussian log-prob / ratio are recomputed here. Episode boundaries (ep_ptr) are
true terminals, so no value bootstrapping crosses episodes.
"""
from __future__ import annotations

import numpy as np

LOG2PI = np.log(2.0 * np.pi)


def gaussian_logp(act, mu, log_std):
    """Per-sample log N(act; mu, exp(log_std)) summed over action dims."""
    std = np.exp(log_std)
    z = (act - mu) / std
    return -0.5 * np.sum(z * z, axis=1) - np.sum(log_std) - 0.5 * act.shape[1] * LOG2PI


def compute_gae(rew, values, done, ep_ptr, gamma, lam):
    """GAE(lambda) advantages + returns; per-episode, terminal = no bootstrap."""
    T = rew.shape[0]
    adv = np.zeros(T)
    for e in range(len(ep_ptr) - 1):
        lo, hi = ep_ptr[e], ep_ptr[e + 1]
        gae = 0.0
        for t in range(hi - 1, lo - 1, -1):
            nonterminal = 0.0 if done[t] else 1.0
            next_v = values[t + 1] if (t + 1 < hi) else 0.0
            delta = rew[t] + gamma * next_v * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
    returns = adv + values
    return adv, returns


def ppo_update(cfg, pol, val, norm, opt_p, opt_v, traj) -> dict:
    """One PPO update over the round's trajectory; mutates pol/val/norm in place."""
    obs = np.asarray(traj["obs"], dtype=np.float64)
    act = np.asarray(traj["act"], dtype=np.float64)
    rew = np.asarray(traj["rew"], dtype=np.float64)
    done = np.asarray(traj["done"], dtype=bool)
    ep_ptr = np.asarray(traj["ep_ptr"], dtype=np.int64)
    T = obs.shape[0]
    if T == 0:
        return {"mean_reward": 0.0, "n_transitions": 0}

    # Update normalizer; old/new log-probs share it (obs are bounded ~[-1,1]).
    norm.update(obs)
    Xn = norm.normalize(obs)

    values = val.value(Xn)
    adv, returns = compute_gae(rew, values, done, ep_ptr, cfg.gamma, cfg.lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    mu_old = pol.mu(Xn)
    logp_old = gaussian_logp(act, mu_old, pol.log_std)

    clip = cfg.clip
    idx_all = np.arange(T)
    mb = max(1, min(cfg.minibatch, T))
    rng = np.random.default_rng(cfg.seed + 1000)

    last_kl = 0.0
    for _ in range(cfg.ppo_epochs):
        rng.shuffle(idx_all)
        for start in range(0, T, mb):
            idx = idx_all[start:start + mb]
            B = idx.shape[0]
            Xb = Xn[idx]
            ab = act[idx]
            Ab = adv[idx]
            Rb = returns[idx]
            lpo = logp_old[idx]

            # ---- policy ----
            mu = pol.mu(Xb)  # caches activations in pol.mlp
            std = np.exp(pol.log_std)
            inv_var = 1.0 / (std * std)
            z = (ab - mu)
            logp = -0.5 * np.sum(z * z * inv_var, axis=1) - np.sum(pol.log_std) - 0.5 * ab.shape[1] * LOG2PI
            ratio = np.exp(logp - lpo)

            # clipped-surrogate gradient mask
            unclipped = ratio * Ab
            clipped = np.clip(ratio, 1 - clip, 1 + clip) * Ab
            use_unclipped = unclipped <= clipped  # min selects unclipped
            # also: clipping only bites when it would reduce the objective
            g_obj = np.where(use_unclipped, ratio * Ab, 0.0)  # d obj / d logp
            dlogp = -g_obj / B  # loss = -mean(obj)

            dmu = dlogp[:, None] * (z * inv_var)  # d loss / d mu
            dW, db = pol.mlp.backward(dmu)

            # log_std gradient: policy ratio term + entropy bonus
            dlogstd_ratio = (dlogp[:, None] * (z * z * inv_var - 1.0)).sum(axis=0)
            dlogstd_ent = -cfg.ent_coef * np.ones_like(pol.log_std)  # -d(ent_coef*entropy)
            dlogstd = dlogstd_ratio + dlogstd_ent

            opt_p.step(pol.mlp.params() + [pol.log_std],
                       pol.mlp.grads_to_list(dW, db) + [dlogstd], max_norm=1.0)
            np.clip(pol.log_std, cfg.logstd_min, cfg.logstd_max, out=pol.log_std)  # exploration floor/ceiling

            # ---- value ----
            vpred = val.value(Xb)
            dv = (cfg.vf_coef * (vpred - Rb) / B)[:, None]
            dWv, dbv = val.mlp.backward(dv)
            opt_v.step(val.mlp.params(), val.mlp.grads_to_list(dWv, dbv), max_norm=1.0)

        # approximate KL for monitoring
        mu_now = pol.mu(Xn)
        logp_now = gaussian_logp(act, mu_now, pol.log_std)
        last_kl = float(np.mean(logp_old - logp_now))

    # per-episode mean reward
    ep_returns = []
    for e in range(len(ep_ptr) - 1):
        ep_returns.append(float(rew[ep_ptr[e]:ep_ptr[e + 1]].sum()))
    return {
        "mean_reward": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "n_transitions": int(T),
        "approx_kl": last_kl,
        "sigma_policy": float(np.exp(pol.log_std).mean()),
    }
