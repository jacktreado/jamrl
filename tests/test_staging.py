"""Gate: node-local scratch staging (write-to-scratch, copy-out-at-end)."""
import os

import numpy as np

from jamrl import policy, postprocess, rollout, staging, storage
from jamrl.config import Config


def _cfg(tmp_path, node_scratch, name="stg", **kw):
    base = dict(N=32, P=1e-3, T_cap=10, n_relax=20, workers=1, episodes_per_worker=2,
                save_hessian="sparse", hidden=(16, 16), backend="numpy",
                node_scratch=node_scratch, campaign_root=str(tmp_path / "persist"),
                name=name, seed=3)
    base.update(kw)
    return Config(**base)


# --------------------------------------------------------------------------- #
# resolve_base
# --------------------------------------------------------------------------- #
def test_resolve_base_disabled_when_empty(tmp_path):
    assert staging.resolve_base(_cfg(tmp_path, "")) is None
    assert staging.enabled(_cfg(tmp_path, "")) is False


def test_resolve_base_uses_cfg_path(tmp_path):
    s = str(tmp_path / "scratch")
    assert staging.resolve_base(_cfg(tmp_path, s)) == s
    assert os.path.isdir(s)  # created


def test_resolve_base_env_overrides_cfg(tmp_path, monkeypatch):
    env_s = str(tmp_path / "env_scratch")
    monkeypatch.setenv("JAMRL_NODE_SCRATCH", env_s)
    assert staging.resolve_base(_cfg(tmp_path, str(tmp_path / "cfg_scratch"))) == env_s


def test_resolve_base_expands_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SCRATCH", str(tmp_path / "viavar"))
    assert staging.resolve_base(_cfg(tmp_path, "$MY_SCRATCH")) == str(tmp_path / "viavar")


def test_resolve_base_unset_var_falls_back(tmp_path):
    assert staging.resolve_base(_cfg(tmp_path, "$JAMRL_NO_SUCH_VAR_XYZ/scratch")) is None


# --------------------------------------------------------------------------- #
# output() context
# --------------------------------------------------------------------------- #
def test_output_disabled_writes_in_place(tmp_path):
    cfg = _cfg(tmp_path, "")
    dst = str(tmp_path / "persist" / "stg" / "x.bin")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with staging.output(dst, cfg) as p:
        assert p == dst
        with open(p, "wb") as f:
            f.write(b"hi")
    assert open(dst, "rb").read() == b"hi"


def test_output_staged_routes_through_scratch(tmp_path):
    scratch = str(tmp_path / "scratch")
    cfg = _cfg(tmp_path, scratch)
    dst = str(tmp_path / "persist" / "stg" / "deep" / "y.bin")  # parent dirs absent
    with staging.output(dst, cfg) as p:
        assert os.path.abspath(scratch) in os.path.abspath(p)  # writing under scratch
        assert p != dst
        with open(p, "wb") as f:
            f.write(b"payload")
        assert not os.path.exists(dst)  # not on persistent until block exits
        local = p
    assert open(dst, "rb").read() == b"payload"  # copied out
    assert not os.path.exists(local)             # scratch copy removed


def test_output_failure_leaves_no_partial_persistent(tmp_path):
    cfg = _cfg(tmp_path, str(tmp_path / "scratch"))
    dst = str(tmp_path / "persist" / "stg" / "z.bin")
    try:
        with staging.output(dst, cfg) as p:
            with open(p, "wb") as f:
                f.write(b"partial")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not os.path.exists(dst)  # nothing copied out on failure


# --------------------------------------------------------------------------- #
# end-to-end parity: staged vs in-place produce identical persistent data
# --------------------------------------------------------------------------- #
def _run_round0(cfg):
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    policy.init_policy_npz(storage.policy_path(camp, 0), hidden=cfg.hidden, seed=cfg.seed)
    rollout.run_rollout(cfg, camp, 0, 0)
    return camp


def test_rollout_staging_matches_in_place(tmp_path):
    plain = _run_round0(_cfg(tmp_path, "", name="plain"))
    staged = _run_round0(_cfg(tmp_path, str(tmp_path / "scratch"), name="staged"))

    # rollout npz identical
    a = storage.read_rollout_npz(storage.rollout_path(plain, 0, 0))
    b = storage.read_rollout_npz(storage.rollout_path(staged, 0, 0))
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k

    # states h5 lands on persistent with identical jammed-state attrs/data
    sa = list(storage.iter_states_h5(storage.states_path(plain, 0, 0)))
    sb = list(storage.iter_states_h5(storage.states_path(staged, 0, 0)))
    assert len(sa) == len(sb)
    for (na, aa, da), (nb, ab, db) in zip(sa, sb):
        assert na == nb
        assert np.array_equal(da["s"], db["s"])
        assert aa["seed"] == ab["seed"] and aa["n_contacts"] == ab["n_contacts"]


def test_postprocess_staging_writes_to_persistent(tmp_path):
    cfg = _cfg(tmp_path, str(tmp_path / "scratch"), name="ppstg")
    camp = _run_round0(cfg)
    row = postprocess.run_postprocess(cfg, camp, 0, shard=0, nshards=1)
    out = os.path.join(storage.analysis_dir(camp, 0), "spectra_shard_000.npz")
    assert os.path.exists(out)  # spectra copied to persistent analysis dir
    if row["n_states"]:
        with np.load(out, allow_pickle=True) as z:
            assert "eig" in z and len(z["eig"]) == row["n_states"]
