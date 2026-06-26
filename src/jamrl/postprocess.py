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


def _collect_states(camp, r):
    out = []
    for wf in sorted(glob.glob(os.path.join(storage.states_dir(camp, r), "worker_*.h5"))):
        for name, attrs, data in storage.iter_states_h5(wf):
            out.append((attrs, data))
    return out


def run_postprocess(cfg, camp, r, shard=0, nshards=1, dos_k=60):
    states = _collect_states(camp, r)
    mine = states[shard::nshards]

    seeds, Bs, Gs, dzs, phis, omegas = [], [], [], [], [], []
    eig_list = []
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

    os.makedirs(storage.analysis_dir(camp, r), exist_ok=True)
    out_npz = os.path.join(storage.analysis_dir(camp, r), f"spectra_shard_{shard:03d}.npz")
    with staging.output(out_npz, cfg) as op:  # node-scratch -> persistent
        with storage.atomic_path(op) as tmp:
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f, seeds=np.asarray(seeds, np.uint64), B=np.asarray(Bs),
                    G=np.asarray(Gs), dz=np.asarray(dzs), phi=np.asarray(phis),
                    omega_star=np.asarray(omegas),
                    eig=np.array(eig_list, dtype=object),
                )

    n = len(mine)
    row = {
        "round": r, "shard": shard, "n_states": n,
        "Bbar": float(np.mean(Bs)) if n else float("nan"),
        "Gbar": float(np.mean(Gs)) if n else float("nan"),
        "dzbar": float(np.mean(dzs)) if n else float("nan"),
        "omega_star": float(np.nanmean(omegas)) if n else float("nan"),
        "shear_stable_frac": float(np.mean([1.0 if g >= -1e-8 else 0.0 for g in Gs])) if n else float("nan"),
    }
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
