"""Storage layer: campaign layout, npz/h5/parquet schemas (plan section 7).

All writers are atomic (write to ``*.tmp`` then ``os.replace``) so a requeued or
killed task never leaves a half-written file.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os

import numpy as np

# ----------------------------------------------------------------------- #
# Campaign layout (plan section 7.1)
# ----------------------------------------------------------------------- #
def campaign_dir(cfg) -> str:
    return os.path.join(cfg.campaign_root, cfg.name)


def _p(*parts) -> str:
    return os.path.join(*parts)


def policy_path(camp, r):       return _p(camp, "policy", f"round_{r:04d}.npz")
def checkpoint_path(camp, r):   return _p(camp, "checkpoints", f"learner_round_{r:04d}.npz")
def rollout_dir(camp, r):       return _p(camp, "rollouts", f"round_{r:04d}")
def rollout_path(camp, r, k):   return _p(rollout_dir(camp, r), f"worker_{k:03d}.npz")
def states_dir(camp, r):        return _p(camp, "states", f"round_{r:04d}")
def states_path(camp, r, k):    return _p(states_dir(camp, r), f"worker_{k:03d}.h5")
def analysis_dir(camp, r):      return _p(camp, "analysis", f"round_{r:04d}")
def summary_parquet(camp):      return _p(camp, "analysis", "summary.parquet")
def round_json(camp, r):        return _p(camp, "rounds", f"round_{r:04d}.json")
def null_cache_dir(camp):       return _p(camp, "null_cache")


def ensure_campaign_dirs(camp):
    for sub in ("policy", "checkpoints", "rollouts", "states", "analysis",
                "rounds", "logs", "null_cache"):
        os.makedirs(_p(camp, sub), exist_ok=True)


# ----------------------------------------------------------------------- #
# Atomic write helpers
# ----------------------------------------------------------------------- #
@contextlib.contextmanager
def atomic_path(path):
    """Yield a temp path; os.replace it onto `path` on clean exit."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.remove(tmp)


@contextlib.contextmanager
def file_lock(path):
    """Exclusive cross-process lock via flock on a sidecar .lock file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lockf = open(path + ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


# ----------------------------------------------------------------------- #
# Trajectory npz (plan section 7.2)
# ----------------------------------------------------------------------- #
def pack_trajectories(episodes: list[dict]) -> dict:
    obs, act, rew, done = [], [], [], []
    ep_ptr = [0]
    seeds, outcome, phi, phi_null, steps = [], [], [], [], []
    for e in episodes:
        T = int(e["T"])
        o = np.asarray(e["obs"], dtype=np.float32).reshape(T, -1)
        a = np.asarray(e["act"], dtype=np.float32).reshape(T, -1)
        r = np.asarray(e["rew"], dtype=np.float32).reshape(T)
        d = np.zeros(T, dtype=bool)
        if T > 0:
            d[-1] = True  # every episode boundary is a true terminal
        obs.append(o); act.append(a); rew.append(r); done.append(d)
        ep_ptr.append(ep_ptr[-1] + T)
        seeds.append(np.uint64(e["seed"]))
        outcome.append(np.int8(e["outcome"]))
        # failed episodes (blowup) can have a non-physical phi; clip for float32.
        phi.append(np.float32(np.clip(e["phi"], -1e6, 1e6)))
        phi_null.append(np.float32(np.clip(e["phi_null"], -1e6, 1e6)))
        steps.append(np.int16(e["steps"]))

    obs_dim = obs[0].shape[1] if obs else 10
    act_dim = act[0].shape[1] if act else 2
    return {
        "obs": np.concatenate(obs) if obs else np.zeros((0, obs_dim), np.float32),
        "act": np.concatenate(act) if act else np.zeros((0, act_dim), np.float32),
        "rew": np.concatenate(rew) if rew else np.zeros((0,), np.float32),
        "done": np.concatenate(done) if done else np.zeros((0,), bool),
        "ep_ptr": np.asarray(ep_ptr, np.int32),
        "seeds": np.asarray(seeds, np.uint64),
        "outcome": np.asarray(outcome, np.int8),
        "phi": np.asarray(phi, np.float32),
        "phi_null": np.asarray(phi_null, np.float32),
        "steps": np.asarray(steps, np.int16),
    }


def write_rollout_npz(path, episodes):
    data = pack_trajectories(episodes)
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)


def read_rollout_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------- #
# Jammed-state HDF5 (plan section 7.2)
# ----------------------------------------------------------------------- #
def write_states_h5(path, episodes, radii, P_target, save_hessian="sparse",
                    z_iso=4.0, compression="gzip"):
    import h5py
    from scipy.sparse import csr_matrix

    jammed = [e for e in episodes if e.get("jammed")]
    radii = np.asarray(radii, dtype=np.float32)
    with atomic_path(path) as tmp:
        with h5py.File(tmp, "w") as f:
            f.attrs["n_jammed"] = len(jammed)
            f.attrs["save_hessian"] = save_hessian
            for j, e in enumerate(jammed):
                g = f.create_group(f"ep{j}")
                N = (e["x_final"].shape[0] - 2) // 2
                s = np.asarray(e["x_final"][: 2 * N], dtype=np.float32).reshape(N, 2)
                g.attrs.update(
                    seed=int(e["seed"]), outcome=int(e["outcome"]),
                    L=float(e["L"]), gamma=float(e["gamma"]),
                    P_target=float(P_target), P_int=float(e["P_int"]),
                    phi=float(e["phi"]), z=float(e["z"]), z_iso=float(e.get("z_iso", z_iso)),
                    dz=float(e["dz"]), n_keep=int(e["n_keep"]),
                    n_rattlers=int(e["n_rattlers"]), n_contacts=int(e["n_contacts"]),
                )
                if "B" in e:
                    g.attrs["B"] = float(e["B"])
                    g.attrs["G"] = float(e["G"])
                g.create_dataset("s", data=s, compression=compression)
                g.create_dataset("radii", data=radii, compression=compression)
                g.create_dataset("contacts", data=np.asarray(e["contacts"], np.int32),
                                 compression=compression)
                # terminal relaxation displacement (2N+2: particle ds, dlnL, dgamma),
                # for projecting the motion into the jammed state onto the relaxation modes.
                if "disp" in e and np.asarray(e["disp"]).size:
                    g.create_dataset("disp", data=np.asarray(e["disp"], np.float32),
                                     compression=compression)

                if save_hessian == "sparse" and "H_data" in e:
                    g.create_dataset("H_data", data=np.asarray(e["H_data"], np.float64),
                                     compression=compression)
                    g.create_dataset("H_indices", data=np.asarray(e["H_indices"], np.int32),
                                     compression=compression)
                    g.create_dataset("H_indptr", data=np.asarray(e["H_indptr"], np.int32),
                                     compression=compression)
                    g.attrs["H_shape"] = np.asarray(e["H_shape"], np.int32)
                elif save_hessian == "dense" and "H_data" in e:
                    shape = tuple(int(x) for x in e["H_shape"])
                    H = csr_matrix((e["H_data"], e["H_indices"], e["H_indptr"]), shape=shape)
                    g.create_dataset("H", data=H.toarray().astype(np.float32),
                                     compression=compression)
                elif save_hessian == "spectrum" and "eig" in e:
                    g.create_dataset("eig", data=np.asarray(e["eig"], np.float32),
                                     compression=compression)


def iter_states_h5(path):
    """Yield (group_name, attrs_dict, data_dict) per jammed episode.

    Datasets are loaded into numpy arrays so consumers remain valid after the
    file is closed.
    """
    import h5py

    def _idx(name):
        try:
            return int(name[2:])  # "ep{j}"
        except ValueError:
            return name

    with h5py.File(path, "r") as f:
        for name in sorted(f.keys(), key=_idx):
            g = f[name]
            data = {k: g[k][()] for k in g.keys()}
            yield name, dict(g.attrs), data


# ----------------------------------------------------------------------- #
# Summary parquet (append-only; plan section 7.2)
# ----------------------------------------------------------------------- #
SUMMARY_COLUMNS = [
    "round", "episodes", "mean_reward", "eval_dphi", "eval_dG", "eval_speed", "eval_success",
    "eval_cost_kevals", "mean_absaP", "mean_absaS", "mean_absgamma",
    "Bbar", "Gbar", "dzbar", "rattler_frac", "shear_stable_frac",
    "omega_star", "sigma_policy", "wall_seconds", "git_hash",
]


def append_summary(camp, row: dict):
    import pandas as pd

    path = summary_parquet(camp)
    full = {c: row.get(c, np.nan) for c in SUMMARY_COLUMNS}
    if "git_hash" in row:
        full["git_hash"] = row["git_hash"]
    with file_lock(path):
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = pd.concat([df, pd.DataFrame([full])], ignore_index=True)
        else:
            df = pd.DataFrame([full])
        with atomic_path(path) as tmp:
            df.to_parquet(tmp, index=False)


def read_summary(camp):
    import pandas as pd

    path = summary_parquet(camp)
    if not os.path.exists(path):
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.read_parquet(path)


# ----------------------------------------------------------------------- #
# Null-baseline cache (sharded one-file-per-key; plan section 5.5)
# ----------------------------------------------------------------------- #
# Each (N, P, seed, field) baseline lives in its own tiny text file under
# ``null_cache/`` and is written atomically (tmp + os.replace). This avoids the
# single shared HDF5 file whose concurrent in-place writes corrupt across nodes:
# fcntl.flock only serializes processes on one kernel, so rollout array tasks
# spread over multiple HPC nodes raced and produced "bad symbol table node
# signature". Rollout workers hold disjoint seeds and eval/CEM run single-
# process, so no two writers ever target the same key file; even if they did,
# os.replace makes last-writer-wins safe. Readers stat individual files, so a
# corrupt or half-written shard degrades to a single recompute, not a crash.
def _null_key(N, P, seed, field="phi"):
    base = f"N{int(N)}_P{float(P):.6e}_s{int(seed)}"
    return base if field == "phi" else f"{base}__{field}"


def _null_shard(camp, N, P, seed, field="phi"):
    return _p(null_cache_dir(camp), _null_key(N, P, seed, field) + ".txt")


def null_cache_get(camp, keys: list[tuple], field: str = "phi") -> dict:
    """Return {(N,P,seed): value} for cached entries among `keys`.

    `field` selects which baseline ("phi" = null density, "G" = null shear
    modulus); non-phi fields use a suffixed filename so caches coexist on disk.
    A missing or unreadable shard is simply omitted (treated as not cached).
    """
    out = {}
    for (N, P, seed) in keys:
        try:
            with open(_null_shard(camp, N, P, seed, field)) as f:
                out[(N, P, seed)] = float(f.read())
        except (FileNotFoundError, ValueError):
            continue
    return out


def null_cache_update(camp, mapping: dict, field: str = "phi"):
    """Write {(N,P,seed): value} as atomic per-key shards (no shared file)."""
    for (N, P, seed), val in mapping.items():
        with atomic_path(_null_shard(camp, N, P, seed, field)) as tmp:
            with open(tmp, "w") as f:
                f.write(repr(float(val)))


# ----------------------------------------------------------------------- #
# Round bookkeeping
# ----------------------------------------------------------------------- #
def write_round_json(camp, r, payload: dict):
    with atomic_path(round_json(camp, r)) as tmp:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)


def read_round_json(camp, r) -> dict:
    path = round_json(camp, r)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
