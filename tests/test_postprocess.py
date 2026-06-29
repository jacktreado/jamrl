"""Phase 10 gate: postprocess -> spectra/moduli -> parquet; plot; compact."""
import glob
import os

import numpy as np

from jamrl import analyze, policy, postprocess, rollout, storage
from jamrl.config import Config


def _round0(tmp_path, save_hessian, name="pp", **extra):
    cfg = Config(N=32, P=1e-3, T_cap=12, n_relax=30, workers=2, episodes_per_worker=3,
                 save_hessian=save_hessian, hidden=(16, 16),
                 campaign_root=str(tmp_path), name=name, seed=2, **extra)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)
    for k in range(cfg.workers):
        rollout.run_rollout(cfg, camp, 0, k)
    return cfg, camp


def test_postprocess_sparse_to_parquet(tmp_path):
    cfg, camp = _round0(tmp_path, "sparse")
    row = postprocess.run_postprocess(cfg, camp, 0, shard=0, nshards=1)
    assert row["n_states"] >= 1
    assert np.isfinite(row["Gbar"]) and np.isfinite(row["omega_star"])
    assert row["shear_stable_frac"] >= 0.0

    npz = glob.glob(os.path.join(storage.analysis_dir(camp, 0), "spectra_shard_000.npz"))
    assert npz
    with np.load(npz[0], allow_pickle=True) as z:
        assert "eig" in z and "G" in z and "omega_star" in z
        assert len(z["eig"]) == row["n_states"]
        assert (z["G"] >= -1e-8).all()  # shear-stabilized

    df = postprocess.read_postprocess(camp)
    assert len(df) >= 1 and "Gbar" in df.columns and "omega_star" in df.columns


def test_postprocess_uses_stored_spectrum(tmp_path):
    cfg, camp = _round0(tmp_path, "spectrum", name="pp_spec")
    # states carry 'eig' directly; postprocess should consume them
    row = postprocess.run_postprocess(cfg, camp, 0, shard=0, nshards=1)
    assert row["n_states"] >= 1
    assert np.isfinite(row["omega_star"])


def test_postprocess_sharding_covers_all(tmp_path):
    cfg, camp = _round0(tmp_path, "spectrum", name="pp_shard")
    total = sum(1 for _ in postprocess._collect_states(camp, 0))
    n0 = postprocess.run_postprocess(cfg, camp, 0, shard=0, nshards=2)["n_states"]
    n1 = postprocess.run_postprocess(cfg, camp, 0, shard=1, nshards=2)["n_states"]
    assert n0 + n1 == total


def test_compact_downconverts_hessians(tmp_path):
    cfg, camp = _round0(tmp_path, "sparse", name="pp_compact")
    wf = sorted(glob.glob(os.path.join(storage.states_dir(camp, 0), "worker_*.h5")))[0]
    before = list(storage.iter_states_h5(wf))
    if before:  # at least one jammed state carries a Hessian
        assert any("H_data" in d for _, _, d in before)
    postprocess.compact(camp, lo=0, hi=0, keep_spectrum=True)
    after = list(storage.iter_states_h5(wf))
    for _, _, d in after:
        assert "H_data" not in d and "H" not in d
        assert "eig" in d  # replaced by spectrum


def test_postprocess_dos_full_and_projection(tmp_path):
    """dos_full adds the box-inclusive spectrum and the relaxation-mode projection."""
    cfg, camp = _round0(tmp_path, "spectrum", name="pp_full", dos_full=True, proj_k=20)
    # disp is persisted per jammed state for the projection
    wf = sorted(glob.glob(os.path.join(storage.states_dir(camp, 0), "worker_*.h5")))[0]
    states = list(storage.iter_states_h5(wf))
    if states:
        assert any("disp" in d for _, _, d in states)

    row = postprocess.run_postprocess(cfg, camp, 0, shard=0, nshards=1)
    assert "omega_box_star" in row and "soft_proj_frac" in row

    npz = glob.glob(os.path.join(storage.analysis_dir(camp, 0), "spectra_shard_000.npz"))[0]
    with np.load(npz, allow_pickle=True) as z:
        assert "eig_full" in z and "proj_omega" in z and "proj_w" in z
        assert "omega_box_star" in z and "soft_proj_frac" in z
        assert len(z["eig_full"]) == row["n_states"]
        # full enthalpy Hessian has 2N+2 modes
        if row["n_states"]:
            assert len(z["eig_full"][0]) == 2 * cfg.N + 2
            # projection weights are normalized fractions in [0, 1]
            for w in z["proj_w"]:
                w = np.asarray(w)
                if w.size:
                    assert w.min() >= -1e-6 and w.sum() <= 1.0 + 1e-6


def test_plot_campaign_writes_png(tmp_path):
    cfg = Config(campaign_root=str(tmp_path), name="plt")
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    for r in range(4):
        storage.append_summary(camp, {
            "round": r, "episodes": 10, "mean_reward": -1.0 + 0.2 * r,
            "eval_dphi": 0.001 * r, "eval_success": 1.0, "Gbar": 0.05,
            "dzbar": 0.1, "mean_absaP": 0.3, "sigma_policy": 0.6,
        })
    outs = analyze.plot_campaign(camp)
    assert outs and os.path.exists(outs[0])
