"""Phase 9 gate: SLURM template rendering + dependency wiring (dry run)."""
import os

from jamrl import slurm, storage
from jamrl.config import Config


def _cfg(tmp_path, **kw):
    base = dict(N=64, workers=4, episodes_per_worker=2, rounds=3,
                campaign_root=str(tmp_path), name="slurm", threads_per_task=8)
    base.update(kw)
    return Config(**base)


def test_submit_dry_run_wires_dependencies(tmp_path):
    cfg = _cfg(tmp_path)
    rc = slurm.submit(cfg, test_only=True)
    assert rc == 0
    camp = storage.campaign_dir(cfg)

    # initial policy + config + provenance created
    assert os.path.exists(storage.policy_path(camp, 0))
    assert os.path.exists(os.path.join(camp, "config.yaml"))
    assert os.path.exists(os.path.join(camp, "provenance.json"))

    rj = storage.read_round_json(camp, 0)
    assert rj["roll_jid"] is not None
    assert rj["learn_jid"] is not None and rj["learn_jid"] != rj["roll_jid"]
    assert rj["post_jid"] is not None and rj["post_jid"] != rj["roll_jid"]
    # learn & post depend on the rollout array
    assert rj["dependency"] == f"afterany:{rj['roll_jid']}"


def test_rendered_scripts_content(tmp_path):
    cfg = _cfg(tmp_path)
    slurm.submit(cfg, test_only=True)
    camp = storage.campaign_dir(cfg)
    roll = open(os.path.join(camp, ".sbatch", "rollout_r0000.sbatch")).read()
    learn = open(os.path.join(camp, ".sbatch", "learn_r0000.sbatch")).read()
    post = open(os.path.join(camp, ".sbatch", "postprocess_r0000.sbatch")).read()

    assert "--array=0-3" in roll  # workers=4 -> 0..3
    assert "--cpus-per-task=8" in roll
    assert "srun jamrl rollout" in roll
    assert "OPENBLAS_NUM_THREADS=1" in roll  # episode mode

    assert "srun jamrl learn" in learn
    assert "--cpus-per-task=4" in learn

    assert "srun jamrl postprocess" in post
    assert "--array=0-" in post


def test_dependency_mode_afterok(tmp_path):
    cfg = _cfg(tmp_path, dependency_mode="afterok")
    slurm.submit(cfg, test_only=True)
    camp = storage.campaign_dir(cfg)
    rj = storage.read_round_json(camp, 0)
    assert rj["dependency"].startswith("afterok:")


def test_maybe_resubmit_perpetuates_then_done(tmp_path):
    cfg = _cfg(tmp_path, rounds=3)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    # mid-campaign: resubmits the next round
    slurm.maybe_resubmit(cfg, camp, 0)
    assert os.path.exists(storage.round_json(camp, 1))
    assert not os.path.exists(os.path.join(camp, "DONE"))
    # last round: writes DONE instead of resubmitting
    slurm.maybe_resubmit(cfg, camp, cfg.rounds - 1)
    assert os.path.exists(os.path.join(camp, "DONE"))


def test_stop_sentinel_halts(tmp_path):
    cfg = _cfg(tmp_path, rounds=10)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    open(os.path.join(camp, "STOP"), "w").close()
    slurm.maybe_resubmit(cfg, camp, 0)
    assert os.path.exists(os.path.join(camp, "DONE"))
    assert not os.path.exists(storage.round_json(camp, 1))  # did not resubmit


def test_resume_round_detection(tmp_path):
    cfg = _cfg(tmp_path)
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    from jamrl import policy
    for r in (0, 1, 2):
        policy.init_policy_npz(storage.policy_path(camp, r), hidden=cfg.hidden, seed=r)
    assert slurm.resume_round(camp) == 2
