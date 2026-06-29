"""Condense a completed campaign into a portable HDF5 for offline notebook analysis.

Run on the cluster after a campaign finishes:
    jamrl analyze --campaign /path/to/campaign

Then scp the resulting campaign_analysis.h5 to your laptop and open the notebooks/.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from jamrl import storage

OBS_NAMES = ["fInf", "P_int", "Egamma", "gamma", "phi", "z", "maxOv", "t", "prev_aP", "prev_aS"]
VDOS_BINS = 100
MECH_BINS = 50
ACT_BINS  = 60
OBS_BINS  = 60


# --------------------------------------------------------------------------- #
# Round discovery helpers
# --------------------------------------------------------------------------- #

def _available_rounds(camp: str) -> list[int]:
    policy_dir = os.path.join(camp, "policy")
    rounds = []
    if os.path.isdir(policy_dir):
        for fn in os.listdir(policy_dir):
            if fn.startswith("round_") and fn.endswith(".npz"):
                try:
                    rounds.append(int(fn[6:10]))
                except ValueError:
                    pass
    return sorted(rounds)


def _strided_rounds(all_rounds: list[int], stride: int) -> list[int]:
    if not all_rounds:
        return []
    selected = [r for r in all_rounds if r % stride == 0]
    if all_rounds[0] not in selected:
        selected.insert(0, all_rounds[0])
    if all_rounds[-1] not in selected:
        selected.append(all_rounds[-1])
    return sorted(set(selected))


# --------------------------------------------------------------------------- #
# Two-pass histogram helper
# --------------------------------------------------------------------------- #

def _two_pass_histograms(
    data_by_round: dict,
    n_bins: int,
    percentile_clip: float = 99.9,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (edges (n_bins+1,), counts (n_rounds, n_bins), rounds (n_rounds,))."""
    rounds_list = sorted(data_by_round.keys())
    if not rounds_list:
        return np.zeros(n_bins + 1, np.float32), np.zeros((0, n_bins), np.float32), np.array([], np.int32)

    if vmin is None or vmax is None:
        all_vals = np.concatenate([np.asarray(data_by_round[r]).ravel() for r in rounds_list])
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            return np.zeros(n_bins + 1, np.float32), np.zeros((len(rounds_list), n_bins), np.float32), np.array(rounds_list, np.int32)
        if vmin is None:
            vmin = float(np.percentile(all_vals, 100 - percentile_clip))
        if vmax is None:
            vmax = float(np.percentile(all_vals, percentile_clip))
    if vmax <= vmin:
        vmax = vmin + 1.0

    edges = np.linspace(vmin, vmax, n_bins + 1, dtype=np.float32)
    counts = np.zeros((len(rounds_list), n_bins), np.float32)
    for i, r in enumerate(rounds_list):
        vals = np.asarray(data_by_round[r]).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            c, _ = np.histogram(vals, bins=edges)
            s = c.sum()
            counts[i] = c / s if s else c
    return edges, counts, np.array(rounds_list, np.int32)


# --------------------------------------------------------------------------- #
# VDOS helper
# --------------------------------------------------------------------------- #

def _eig_to_omega(eig, drop_below: float = 1e-8) -> np.ndarray:
    omega = np.sqrt(np.clip(np.asarray(eig, np.float64), 0.0, None))
    return omega[omega > drop_below]


# --------------------------------------------------------------------------- #
# Section: summary
# --------------------------------------------------------------------------- #

def _build_summary(camp: str) -> dict:
    print("[analyze] summary: reading parquet ...", flush=True)
    df = storage.read_summary(camp)
    if df.empty:
        return {}
    cols = [
        "round", "episodes", "mean_reward", "eval_dphi", "eval_dG", "eval_success",
        "mean_absaP", "mean_absaS", "mean_absgamma",
        "Bbar", "Gbar", "dzbar", "rattler_frac", "shear_stable_frac",
        "omega_star", "sigma_policy", "wall_seconds",
    ]
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = df[c].to_numpy(dtype=np.float32 if c != "round" else np.int32)
        else:
            n = len(df)
            out[c] = np.full(n, np.nan, dtype=np.float32) if c != "round" else np.zeros(n, np.int32)
    print(f"[analyze] summary: {len(df)} rounds", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Section: policy
# --------------------------------------------------------------------------- #

def _build_policy(camp: str, rounds: list[int]) -> dict:
    from jamrl import policy as pol_mod
    print(f"[analyze] policy: loading {len(rounds)} rounds ...", flush=True)

    log_std_list, obs_mean_list, obs_std_list = [], [], []
    W0_list, W1_list, Wmu_list = [], [], []
    W0_norm, W1_norm, Wmu_norm = [], [], []
    valid_rounds = []

    for r in rounds:
        path = storage.policy_path(camp, r)
        if not os.path.exists(path):
            continue
        try:
            d = pol_mod.load_policy_npz(path)
        except Exception:
            continue
        valid_rounds.append(r)
        log_std_list.append(d["log_std"].astype(np.float32))
        obs_mean_list.append(d["obs_mean"].astype(np.float32))
        obs_std_list.append(d["obs_std"].astype(np.float32))
        W0 = d["W0"].astype(np.float32)
        W1 = d["W1"].astype(np.float32)
        Wmu = d["Wmu"].astype(np.float32)
        W0_list.append(W0); W1_list.append(W1); Wmu_list.append(Wmu)
        W0_norm.append(float(np.linalg.norm(W0, "fro")))
        W1_norm.append(float(np.linalg.norm(W1, "fro")))
        Wmu_norm.append(float(np.linalg.norm(Wmu, "fro")))

    if not valid_rounds:
        return {}

    print(f"[analyze] policy: loaded {len(valid_rounds)} rounds", flush=True)
    return {
        "rounds":   np.array(valid_rounds, np.int32),
        "log_std":  np.stack(log_std_list),
        "obs_mean": np.stack(obs_mean_list),
        "obs_std":  np.stack(obs_std_list),
        "W0_norm":  np.array(W0_norm, np.float32),
        "W1_norm":  np.array(W1_norm, np.float32),
        "Wmu_norm": np.array(Wmu_norm, np.float32),
        "W0":       np.stack(W0_list),
        "W1":       np.stack(W1_list),
        "Wmu":      np.stack(Wmu_list),
    }


# --------------------------------------------------------------------------- #
# Section: VDOS + mechanics
# --------------------------------------------------------------------------- #

def _collect_spectra_round(camp: str, r: int) -> dict | None:
    adir = storage.analysis_dir(camp, r)
    shards = sorted(glob.glob(os.path.join(adir, "spectra_shard_*.npz")))
    if not shards:
        return None
    Bs, Gs, dzs, phis, omegas, eig_list = [], [], [], [], [], []
    # box-inclusive spectrum + relaxation-mode projection (plan parts B/C)
    eig_full_list, omega_box, proj_om_list, proj_w_list, soft_fracs = [], [], [], [], []
    for sf in shards:
        try:
            z = np.load(sf, allow_pickle=True)
        except Exception:
            continue
        if "B" in z: Bs.extend(z["B"].tolist())
        if "G" in z: Gs.extend(z["G"].tolist())
        if "dz" in z: dzs.extend(z["dz"].tolist())
        if "phi" in z: phis.extend(z["phi"].tolist())
        if "omega_star" in z: omegas.extend(z["omega_star"].tolist())
        if "eig" in z:
            for e in z["eig"]:
                eig_list.append(np.asarray(e, np.float32))
        if "eig_full" in z:
            for e in z["eig_full"]:
                eig_full_list.append(np.asarray(e, np.float32))
        if "omega_box_star" in z: omega_box.extend(z["omega_box_star"].tolist())
        if "proj_omega" in z:
            for e in z["proj_omega"]:
                proj_om_list.append(np.asarray(e, np.float32))
        if "proj_w" in z:
            for e in z["proj_w"]:
                proj_w_list.append(np.asarray(e, np.float32))
        if "soft_proj_frac" in z: soft_fracs.extend(z["soft_proj_frac"].tolist())
    if not phis:
        return None
    return {
        "B": np.array(Bs, np.float32),
        "G": np.array(Gs, np.float32),
        "dz": np.array(dzs, np.float32),
        "phi": np.array(phis, np.float32),
        "omega_star": np.array(omegas, np.float32),
        "eig_list": eig_list,
        "eig_full_list": eig_full_list,
        "omega_box_star": np.array(omega_box, np.float32),
        "proj_omega_list": proj_om_list,
        "proj_w_list": proj_w_list,
        "soft_proj_frac": np.array(soft_fracs, np.float32),
    }


def _build_vdos_and_mechanics(camp: str, spectra_rounds: list[int]) -> dict:
    print(f"[analyze] vdos+mechanics: scanning {len(spectra_rounds)} rounds ...", flush=True)

    per_round: dict[int, dict] = {}
    for r in spectra_rounds:
        d = _collect_spectra_round(camp, r)
        if d is not None:
            per_round[r] = d

    if not per_round:
        print("[analyze] vdos+mechanics: no spectra found", flush=True)
        return {}

    valid = sorted(per_round.keys())
    print(f"[analyze] vdos+mechanics: {len(valid)} rounds with spectra", flush=True)

    # ---- VDOS ----
    # pass 1: find global omega_max
    omega_max = 0.0
    for r in valid:
        for eig in per_round[r]["eig_list"]:
            om = _eig_to_omega(eig)
            if om.size:
                omega_max = max(omega_max, float(np.percentile(om, 99.9)))
    if omega_max == 0.0:
        omega_max = 1.0

    omega_edges = np.linspace(0.0, omega_max, VDOS_BINS + 1, dtype=np.float32)
    vdos_counts = np.zeros((len(valid), VDOS_BINS), np.float32)
    vdos_omega_star = np.zeros(len(valid), np.float32)
    vdos_n_states = np.zeros(len(valid), np.int32)

    for i, r in enumerate(valid):
        d = per_round[r]
        all_omega = np.concatenate([_eig_to_omega(e) for e in d["eig_list"]]) if d["eig_list"] else np.array([])
        vdos_n_states[i] = len(d["eig_list"])
        vdos_omega_star[i] = float(np.nanmean(d["omega_star"])) if d["omega_star"].size else np.nan
        if all_omega.size:
            c, _ = np.histogram(all_omega, bins=omega_edges)
            s = c.sum()
            vdos_counts[i] = c / s if s else c

    # ---- Mechanics (two-pass histograms) ----
    mech_keys = ["B", "G", "dz", "phi"]
    mech_data: dict[str, dict[int, np.ndarray]] = {k: {} for k in mech_keys}
    mech_mean: dict[str, list] = {k: [] for k in mech_keys}
    mech_std:  dict[str, list] = {k: [] for k in mech_keys}

    for r in valid:
        d = per_round[r]
        for k in mech_keys:
            arr = d[k]
            mech_data[k][r] = arr
            mech_mean[k].append(float(np.nanmean(arr)) if arr.size else np.nan)
            mech_std[k].append(float(np.nanstd(arr))  if arr.size else np.nan)

    mech_edges: dict[str, np.ndarray] = {}
    mech_counts: dict[str, np.ndarray] = {}
    for k in mech_keys:
        edges, counts, _ = _two_pass_histograms(mech_data[k], MECH_BINS)
        mech_edges[k] = edges
        mech_counts[k] = counts

    out = {
        "vdos": {
            "rounds":      np.array(valid, np.int32),
            "omega_edges": omega_edges,
            "counts":      vdos_counts,
            "omega_star":  vdos_omega_star,
            "n_states":    vdos_n_states,
        },
        "mechanics": {
            "rounds":         np.array(valid, np.int32),
            "n_states":       vdos_n_states,
            **{f"{k}_hist_edges": mech_edges[k] for k in mech_keys},
            **{f"{k}_counts":     mech_counts[k] for k in mech_keys},
            **{f"{k}_mean": np.array(mech_mean[k], np.float32) for k in mech_keys},
            **{f"{k}_std":  np.array(mech_std[k],  np.float32) for k in mech_keys},
        },
    }

    # ---- box-inclusive VDOS + relaxation-mode projection (plan parts B/C) ----
    has_box = any(per_round[r]["eig_full_list"] for r in valid)
    if has_box:
        box_max = 0.0
        for r in valid:
            for eig in per_round[r]["eig_full_list"]:
                om = _eig_to_omega(eig)
                if om.size:
                    box_max = max(box_max, float(np.percentile(om, 99.9)))
        if box_max == 0.0:
            box_max = 1.0
        box_edges = np.linspace(0.0, box_max, VDOS_BINS + 1, dtype=np.float32)
        box_counts = np.zeros((len(valid), VDOS_BINS), np.float32)
        box_omega_star = np.zeros(len(valid), np.float32)
        box_n_states = np.zeros(len(valid), np.int32)
        # projections: average per-mode projection weight as a function of omega
        proj_counts = np.zeros((len(valid), VDOS_BINS), np.float32)
        soft_mean = np.zeros(len(valid), np.float32)

        for i, r in enumerate(valid):
            d = per_round[r]
            box_n_states[i] = len(d["eig_full_list"])
            obs = d["omega_box_star"]
            box_omega_star[i] = float(np.nanmean(obs)) if obs.size else np.nan
            all_box = (np.concatenate([_eig_to_omega(e) for e in d["eig_full_list"]])
                       if d["eig_full_list"] else np.array([]))
            if all_box.size:
                c, _ = np.histogram(all_box, bins=box_edges)
                s = c.sum()
                box_counts[i] = c / s if s else c
            # projection-weight spectrum: histogram omega weighted by projection weight,
            # averaged over states (the two ~zero modes are pre-dropped in postprocess).
            pw = np.zeros(VDOS_BINS, np.float32)
            npos = 0
            for om, wt in zip(d["proj_omega_list"], d["proj_w_list"]):
                om = np.asarray(om, np.float64); wt = np.asarray(wt, np.float64)
                if om.size and wt.size:
                    h, _ = np.histogram(om, bins=box_edges, weights=wt)
                    pw += h.astype(np.float32)
                    npos += 1
            proj_counts[i] = pw / npos if npos else pw
            sf = d["soft_proj_frac"]
            soft_mean[i] = float(np.nanmean(sf)) if sf.size else np.nan

        out["vdos_box"] = {
            "rounds":      np.array(valid, np.int32),
            "omega_edges": box_edges,
            "counts":      box_counts,
            "omega_star":  box_omega_star,
            "n_states":    box_n_states,
        }
        out["projections"] = {
            "rounds":         np.array(valid, np.int32),
            "omega_edges":    box_edges,
            "weight":         proj_counts,   # mean projection weight per omega bin
            "soft_proj_frac": soft_mean,     # motion fraction in softest decile of modes
            "n_states":       box_n_states,
        }

    return out


# --------------------------------------------------------------------------- #
# Section: actions + observations
# --------------------------------------------------------------------------- #

def _load_round_rollouts(camp: str, r: int) -> dict | None:
    rdir = storage.rollout_dir(camp, r)
    files = sorted(glob.glob(os.path.join(rdir, "worker_*.npz")))
    if not files:
        return None
    obs_list, act_list, rew_list = [], [], []
    phi_list, phi_null_list, outcome_list, steps_list = [], [], [], []
    ep_ptr = [0]
    for wf in files:
        try:
            d = storage.read_rollout_npz(wf)
        except Exception:
            continue
        obs_list.append(d["obs"])
        act_list.append(d["act"])
        rew_list.append(d["rew"])
        phi_list.append(d["phi"])
        phi_null_list.append(d["phi_null"])
        outcome_list.append(d["outcome"])
        steps_list.append(d["steps"])
        # merge ep_ptr: offset by current total transitions
        offset = ep_ptr[-1]
        wp = np.asarray(d["ep_ptr"], np.int32)
        ep_ptr.extend((wp[1:] + offset).tolist())
    if not obs_list:
        return None
    return {
        "obs":      np.concatenate(obs_list, axis=0),
        "act":      np.concatenate(act_list, axis=0),
        "rew":      np.concatenate(rew_list, axis=0),
        "ep_ptr":   np.array(ep_ptr, np.int32),
        "phi":      np.concatenate(phi_list),
        "phi_null": np.concatenate(phi_null_list),
        "outcome":  np.concatenate(outcome_list),
        "steps":    np.concatenate(steps_list),
    }


def _build_actions_obs(camp: str, traj_rounds: list[int]) -> dict:
    print(f"[analyze] actions+obs: scanning {len(traj_rounds)} rounds ...", flush=True)

    act_data: dict[str, dict] = {"aP": {}, "aS": {}}
    obs_data: dict[str, dict] = {name: {} for name in OBS_NAMES}
    aP_mean, aP_std, aS_mean, aS_std, n_trans = [], [], [], [], []
    valid = []

    for r in traj_rounds:
        d = _load_round_rollouts(camp, r)
        if d is None:
            continue
        valid.append(r)
        act = d["act"]
        obs = d["obs"]
        act_data["aP"][r] = act[:, 0]
        act_data["aS"][r] = act[:, 1]
        for i, name in enumerate(OBS_NAMES):
            obs_data[name][r] = obs[:, i]
        aP_mean.append(float(np.mean(act[:, 0])))
        aP_std.append(float(np.std(act[:, 0])))
        aS_mean.append(float(np.mean(act[:, 1])))
        aS_std.append(float(np.std(act[:, 1])))
        n_trans.append(len(act))

    if not valid:
        return {}

    print(f"[analyze] actions+obs: {len(valid)} rounds", flush=True)

    aP_edges, aP_counts, _ = _two_pass_histograms(act_data["aP"], ACT_BINS, vmin=-1.05, vmax=1.05)
    aS_edges, aS_counts, _ = _two_pass_histograms(act_data["aS"], ACT_BINS, vmin=-1.05, vmax=1.05)

    obs_edges: dict[str, np.ndarray] = {}
    obs_counts: dict[str, np.ndarray] = {}
    for name in OBS_NAMES:
        edges, counts, _ = _two_pass_histograms(obs_data[name], OBS_BINS)
        obs_edges[name] = edges
        obs_counts[name] = counts

    vrounds = np.array(valid, np.int32)
    return {
        "actions": {
            "rounds":        vrounds,
            "aP_hist_edges": aP_edges,
            "aS_hist_edges": aS_edges,
            "aP_counts":     aP_counts,
            "aS_counts":     aS_counts,
            "aP_mean":       np.array(aP_mean, np.float32),
            "aP_std":        np.array(aP_std,  np.float32),
            "aS_mean":       np.array(aS_mean, np.float32),
            "aS_std":        np.array(aS_std,  np.float32),
            "n_transitions": np.array(n_trans, np.int32),
        },
        "observations": {
            "rounds": vrounds,
            **{f"{name}_edges":  obs_edges[name]  for name in OBS_NAMES},
            **{f"{name}_counts": obs_counts[name] for name in OBS_NAMES},
        },
    }


# --------------------------------------------------------------------------- #
# Section: trajectories
# --------------------------------------------------------------------------- #

def _build_trajectories(camp: str, traj_rounds: list[int]) -> dict:
    print(f"[analyze] trajectories: storing raw data for {len(traj_rounds)} rounds ...", flush=True)
    result = {}
    valid = []
    for r in traj_rounds:
        d = _load_round_rollouts(camp, r)
        if d is None:
            continue
        result[r] = {
            "obs":      d["obs"].astype(np.float32),
            "act":      d["act"].astype(np.float32),
            "rew":      d["rew"].astype(np.float32),
            "ep_ptr":   d["ep_ptr"].astype(np.int32),
            "phi":      np.asarray(d["phi"],      np.float32),
            "phi_null": np.asarray(d["phi_null"], np.float32),
            "outcome":  np.asarray(d["outcome"],  np.int8),
            "steps":    np.asarray(d["steps"],    np.int16),
        }
        valid.append(r)
    print(f"[analyze] trajectories: {len(valid)} rounds stored", flush=True)
    return {"rounds": np.array(valid, np.int32), "per_round": result}


# --------------------------------------------------------------------------- #
# H5 write helpers
# --------------------------------------------------------------------------- #

def _write_arrays(grp, data: dict, compression="gzip"):
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.dtype.kind in ("U", "S", "O"):
            grp.attrs[k] = str(v)
        elif arr.ndim == 0:
            grp.attrs[k] = arr.item()
        else:
            grp.create_dataset(k, data=arr, compression=compression if arr.size > 256 else None)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def build_campaign_analysis(
    camp: str,
    spectra_stride: int = 10,
    traj_stride: int = 25,
    out_path: str | None = None,
) -> str:
    import h5py
    from jamrl.config import Config

    if out_path is None:
        out_path = os.path.join(camp, "analysis", "campaign_analysis.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cfg = None
    cfg_path = os.path.join(camp, "config.yaml")
    if os.path.exists(cfg_path):
        cfg = Config.from_yaml(cfg_path)

    all_rounds = _available_rounds(camp)
    spectra_rounds = _strided_rounds(all_rounds, spectra_stride)
    traj_rounds    = _strided_rounds(all_rounds, traj_stride)

    print(f"[analyze] campaign={camp}", flush=True)
    print(f"[analyze] {len(all_rounds)} policy rounds found; "
          f"{len(spectra_rounds)} spectra rounds; {len(traj_rounds)} traj rounds", flush=True)

    summary   = _build_summary(camp)
    policy    = _build_policy(camp, all_rounds)
    vm        = _build_vdos_and_mechanics(camp, spectra_rounds)
    ao        = _build_actions_obs(camp, traj_rounds)
    trajs     = _build_trajectories(camp, traj_rounds)

    print(f"[analyze] writing {out_path} ...", flush=True)
    with storage.atomic_path(out_path) as tmp:
        with h5py.File(tmp, "w") as f:
            # file-level attrs
            f.attrs["campaign_name"]  = os.path.basename(camp)
            f.attrs["N"]              = int(cfg.N)         if cfg else -1
            f.attrs["P"]              = float(cfg.P)       if cfg else float("nan")
            f.attrs["hidden"]         = list(cfg.hidden)   if cfg else []
            f.attrs["rounds_total"]   = int(cfg.rounds)    if cfg else len(all_rounds)
            f.attrs["reward_mode"]    = cfg.reward_mode    if cfg else "density"
            f.attrs["spectra_stride"] = spectra_stride
            f.attrs["traj_stride"]    = traj_stride

            # summary
            if summary:
                sg = f.create_group("summary")
                _write_arrays(sg, summary)

            # policy
            if policy:
                pg = f.create_group("policy")
                _write_arrays(pg, policy)

            # vdos
            if "vdos" in vm:
                vg = f.create_group("vdos")
                _write_arrays(vg, vm["vdos"])

            # mechanics
            if "mechanics" in vm:
                mg = f.create_group("mechanics")
                _write_arrays(mg, vm["mechanics"])

            # box-inclusive VDOS (full enthalpy Hessian spectrum)
            if "vdos_box" in vm:
                vbg = f.create_group("vdos_box")
                _write_arrays(vbg, vm["vdos_box"])

            # relaxation-mode projections (terminal motion onto full-Hessian modes)
            if "projections" in vm:
                pjg = f.create_group("projections")
                _write_arrays(pjg, vm["projections"])

            # actions
            if "actions" in ao:
                ag = f.create_group("actions")
                _write_arrays(ag, ao["actions"])

            # observations
            if "observations" in ao:
                og = f.create_group("observations")
                og.attrs["obs_names"] = OBS_NAMES
                _write_arrays(og, ao["observations"])

            # trajectories
            if trajs and "per_round" in trajs:
                tg = f.create_group("trajectories")
                tg.create_dataset("rounds", data=trajs["rounds"])
                for r, rd in trajs["per_round"].items():
                    rg = tg.create_group(f"round_{r:04d}")
                    _write_arrays(rg, rd)

    print(f"[analyze] done. File size: {os.path.getsize(out_path) / 1e6:.1f} MB", flush=True)
    return out_path
