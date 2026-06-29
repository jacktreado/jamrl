"""Post-processing: diagonalize Hessians -> DOS spectra + moduli (plan 6.5/10).

Non-blocking: reads only the saved jammed states, so it never stalls training.
Sharded over the round's states for throughput; each shard writes a spectra npz
and a partial aggregate row to analysis/postprocess.parquet.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from jamrl import _core, staging, storage


def _system_from_state(N, P, phi0, s, L, gamma):
    sys = _core.make_system(N, 1, phi0, P)  # radii are deterministic in N
    x = np.concatenate([np.asarray(s, float).ravel(), [np.log(float(L))], [float(gamma)]])
    sys.x = x
    sys.P = float(P)
    return sys


def _omega_star(eig):
    omega = np.sqrt(np.clip(np.asarray(eig, float), 0.0, None))
    nz = omega[omega > 1e-8]  # drop the two ~zero translational modes
    return float(np.percentile(nz, 5)) if nz.size else float("nan")


def project_disp(sys, disp, k):
    """Project the terminal relaxation displacement onto the full-Hessian modes.

    Returns (omega, weight) where omega = sqrt(max(lambda,0)) for the computed
    (lowest-k) relaxation modes and weight = c**2 normalized by the total motion
    magnitude |disp|**2, so weight sums to <=1 (the fraction of the relaxation
    motion captured by each mode). `disp` and the eigenvectors share the 2N+2
    coordinate basis (particle ds, dlnL, dgamma).
    """
    disp = np.asarray(disp, float).ravel()
    w, V = _core.eigvecs_full(sys, None if k <= 0 else k)
    w = np.asarray(w, float)
    V = np.asarray(V, float)
    c = V.T @ disp
    norm2 = float(disp @ disp)
    weight = (c * c) / norm2 if norm2 > 0.0 else np.zeros_like(c)
    omega = np.sqrt(np.clip(w, 0.0, None))
    return omega, weight


def _soft_proj_frac(omega, weight, frac=0.1):
    """Fraction of the relaxation motion captured by the softest `frac` of modes
    (modes are ascending in omega; the two ~zero translational modes are dropped)."""
    nz = omega > 1e-8
    om, wt = omega[nz], weight[nz]
    if om.size == 0:
        return float("nan")
    nsoft = max(1, int(round(om.size * frac)))
    return float(wt[:nsoft].sum())  # eigvecs_full returns modes ascending in lambda


def _collect_states(camp, r):
    out = []
    for wf in sorted(glob.glob(os.path.join(storage.states_dir(camp, r), "worker_*.h5"))):
        for name, attrs, data in storage.iter_states_h5(wf):
            out.append((attrs, data))
    return out


def run_postprocess(cfg, camp, r, shard=0, nshards=1, dos_k=60):
    states = _collect_states(camp, r)
    mine = states[shard::nshards]

    dos_full = bool(getattr(cfg, "dos_full", False))
    proj_k = int(getattr(cfg, "proj_k", 60))

    seeds, Bs, Gs, dzs, phis, omegas = [], [], [], [], [], []
    eig_list = []
    # box-inclusive spectrum + relaxation-mode projection (plan parts B/C)
    eig_full_list, omega_box = [], []
    proj_omega_list, proj_w_list, soft_fracs = [], [], []
    for attrs, data in mine:
        N = int(len(data["radii"]))
        sys = _system_from_state(N, attrs["P_target"], cfg.phi0, data["s"], attrs["L"], attrs["gamma"])

        if "eig" in data:
            eig = np.asarray(data["eig"], float)
        else:
            k = -1 if N <= 512 else dos_k  # dense for small N, Spectra lowest-k otherwise
            eig = np.asarray(_core.eigvals_dos(sys, None if k < 0 else k), float)

        B = float(attrs["B"]) if "B" in attrs else float(_core.bulk_modulus(sys))
        G = float(attrs["G"]) if "G" in attrs else float(_core.shear_modulus(sys))

        seeds.append(int(attrs["seed"]))
        Bs.append(B); Gs.append(G); dzs.append(float(attrs["dz"]))
        phis.append(float(attrs["phi"])); omegas.append(_omega_star(eig))
        eig_list.append(eig.astype(np.float32))

        if dos_full:
            kf = -1 if N <= 512 else proj_k  # dense for small N, Spectra lowest-k otherwise
            eig_f = np.asarray(_core.eigvals_full(sys, None if kf < 0 else kf), float)
            eig_full_list.append(eig_f.astype(np.float32))
            omega_box.append(_omega_star(eig_f))
            if "disp" in data:
                om, wt = project_disp(sys, data["disp"], proj_k)
                proj_omega_list.append(om.astype(np.float32))
                proj_w_list.append(wt.astype(np.float32))
                soft_fracs.append(_soft_proj_frac(om, wt))
            else:
                proj_omega_list.append(np.zeros(0, np.float32))
                proj_w_list.append(np.zeros(0, np.float32))
                soft_fracs.append(float("nan"))

    os.makedirs(storage.analysis_dir(camp, r), exist_ok=True)
    out_npz = os.path.join(storage.analysis_dir(camp, r), f"spectra_shard_{shard:03d}.npz")
    save_kw = dict(
        seeds=np.asarray(seeds, np.uint64), B=np.asarray(Bs),
        G=np.asarray(Gs), dz=np.asarray(dzs), phi=np.asarray(phis),
        omega_star=np.asarray(omegas),
        eig=np.array(eig_list, dtype=object),
    )
    if dos_full:
        save_kw.update(
            eig_full=np.array(eig_full_list, dtype=object),
            omega_box_star=np.asarray(omega_box),
            proj_omega=np.array(proj_omega_list, dtype=object),
            proj_w=np.array(proj_w_list, dtype=object),
            soft_proj_frac=np.asarray(soft_fracs),
        )
    with staging.output(out_npz, cfg) as op:  # node-scratch -> persistent
        with storage.atomic_path(op) as tmp:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **save_kw)

    n = len(mine)
    row = {
        "round": r, "shard": shard, "n_states": n,
        "Bbar": float(np.mean(Bs)) if n else float("nan"),
        "Gbar": float(np.mean(Gs)) if n else float("nan"),
        "dzbar": float(np.mean(dzs)) if n else float("nan"),
        "omega_star": float(np.nanmean(omegas)) if n else float("nan"),
        "shear_stable_frac": float(np.mean([1.0 if g >= -1e-8 else 0.0 for g in Gs])) if n else float("nan"),
    }
    if dos_full:
        row["omega_box_star"] = float(np.nanmean(omega_box)) if omega_box else float("nan")
        row["soft_proj_frac"] = float(np.nanmean(soft_fracs)) if soft_fracs else float("nan")
    _append_postprocess(camp, row)
    print(f"[postprocess] round {r} shard {shard}/{nshards}: {n} states -> {out_npz}")
    return row


def _append_postprocess(camp, row):
    import pandas as pd

    path = os.path.join(camp, "analysis", "postprocess.parquet")
    with storage.file_lock(path):
        if os.path.exists(path):
            df = pd.concat([pd.read_parquet(path), pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        with storage.atomic_path(path) as tmp:
            df.to_parquet(tmp, index=False)


def read_postprocess(camp):
    import pandas as pd

    path = os.path.join(camp, "analysis", "postprocess.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


# ----------------------------------------------------------------------- #
def compact(camp, lo=0, hi=None, keep_spectrum=True):
    """Down-convert full Hessians in old rounds to backbone spectra (plan 7.3)."""
    import h5py

    cfg = None
    cpath = os.path.join(camp, "config.yaml")
    if os.path.exists(cpath):
        from jamrl.config import Config
        cfg = Config.from_yaml(cpath)
    phi0 = cfg.phi0 if cfg else 0.80

    rounds = []
    sdir = os.path.join(camp, "states")
    if os.path.isdir(sdir):
        for d in os.listdir(sdir):
            if d.startswith("round_"):
                r = int(d.split("_")[1])
                if r >= lo and (hi is None or r <= hi):
                    rounds.append(r)
    converted = 0
    for r in sorted(rounds):
        for wf in sorted(glob.glob(os.path.join(storage.states_dir(camp, r), "worker_*.h5"))):
            changed = False
            states = list(storage.iter_states_h5(wf))
            # rewrite the file, replacing H_* / H with eig
            with storage.atomic_path(wf) as tmp:
                with h5py.File(tmp, "w") as f:
                    for name, attrs, data in states:
                        g = f.create_group(name)
                        for k, v in attrs.items():
                            g.attrs[k] = v
                        for ds in ("s", "radii", "contacts"):
                            if ds in data:
                                g.create_dataset(ds, data=data[ds], compression="gzip")
                        if keep_spectrum:
                            if "eig" in data:
                                eig = data["eig"]
                            elif "H_data" in data or "H" in data:
                                N = int(len(data["radii"]))
                                sys = _system_from_state(N, attrs["P_target"], phi0,
                                                         data["s"], attrs["L"], attrs["gamma"])
                                eig = np.asarray(_core.eigvals_dos(sys, None), np.float32)
                                changed = True
                            else:
                                eig = None
                            if eig is not None:
                                g.create_dataset("eig", data=np.asarray(eig, np.float32),
                                                 compression="gzip")
            converted += int(changed)
    print(f"[compact] processed rounds {lo}..{hi}; converted {converted} worker files to spectra")
    return converted
