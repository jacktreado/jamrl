"""SLURM orchestration: render sbatch, wire dependencies, self-perpetuate (plan 6).

Per round r:  rollout array -> (learn, postprocess) both depending on the array
via afterany (robust) or afterok (strict). The learn job resubmits round r+1 at
its end, so one `jamrl submit` launches the whole campaign.

On a host without `sbatch` (or with --test-only) this runs in *dry* mode: it
still renders every script and records the dependency wiring, so the graph can
be validated without queueing work.
"""
from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess

from jamrl import policy, seeding, storage

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_fake_counter = itertools.count(900001)


def _env():
    import jinja2

    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render(template_name: str, ctx: dict) -> str:
    return _env().get_template(template_name).render(**ctx)


def _post_shards(cfg) -> int:
    return max(1, min(8, cfg.workers))


def ctx_rollout(cfg, camp, r):
    return dict(round=r, workers_minus_1=cfg.workers - 1, threads_per_task=cfg.threads_per_task,
                mem_rollout=cfg.mem_rollout, time_rollout=cfg.time_rollout,
                partition=cfg.partition, account=cfg.account, campaign=camp,
                parallel_mode=cfg.parallel_mode, node_scratch=cfg.node_scratch)


def ctx_learn(cfg, camp, r):
    return dict(round=r, mem_learn=cfg.mem_learn, time_learn=cfg.time_learn,
                partition=cfg.partition, account=cfg.account, campaign=camp)


def ctx_post(cfg, camp, r):
    ns = _post_shards(cfg)
    return dict(round=r, nshards=ns, nshards_minus_1=ns - 1, post_cpus=4,
                mem_post=cfg.mem_post, time_post=cfg.time_post,
                partition=cfg.partition, account=cfg.account, campaign=camp,
                node_scratch=cfg.node_scratch)


def parse_jobid(text: str):
    m = re.search(r"Submitted batch job (\d+)", text)
    return int(m.group(1)) if m else None


def _write_script(camp, r, kind, text) -> str:
    d = os.path.join(camp, ".sbatch")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{kind}_r{r:04d}.sbatch")
    with open(path, "w") as f:
        f.write(text)
    return path


def _sbatch(script_path, extra_args, dry, test_only):
    """Submit (or dry-simulate) one script; return a job id (real or fake)."""
    if dry:
        if test_only and shutil.which("sbatch"):
            # validate the script without queueing (ignore output)
            subprocess.run(["sbatch", "--test-only", *extra_args, script_path],
                           capture_output=True, text=True)
        return next(_fake_counter)
    out = subprocess.check_output(["sbatch", *extra_args, script_path], text=True)
    return parse_jobid(out)


def submit_round(cfg, camp, r, dry=False, test_only=False) -> dict:
    roll_txt = render("rollout.sbatch.j2", ctx_rollout(cfg, camp, r))
    learn_txt = render("learn.sbatch.j2", ctx_learn(cfg, camp, r))
    post_txt = render("postprocess.sbatch.j2", ctx_post(cfg, camp, r))
    roll_p = _write_script(camp, r, "rollout", roll_txt)
    learn_p = _write_script(camp, r, "learn", learn_txt)
    post_p = _write_script(camp, r, "postprocess", post_txt)

    dep = cfg.dependency_mode  # afterany | afterok
    roll_jid = _sbatch(roll_p, [], dry, test_only)
    dep_arg = [f"--dependency={dep}:{roll_jid}"] if roll_jid is not None else []
    learn_jid = _sbatch(learn_p, dep_arg, dry, test_only)
    post_jid = _sbatch(post_p, dep_arg, dry, test_only)

    payload = {
        "round": r,
        "roll_jid": roll_jid,
        "learn_jid": learn_jid,
        "post_jid": post_jid,
        "dependency": f"{dep}:{roll_jid}",
        "dry": dry,
    }
    storage.write_round_json(camp, r, payload)
    return payload


def resume_round(camp) -> int:
    """Highest round r whose policy/round_r.npz exists (where to (re)start)."""
    pdir = os.path.join(camp, "policy")
    best = 0
    if os.path.isdir(pdir):
        for fn in os.listdir(pdir):
            m = re.match(r"round_(\d+)\.npz$", fn)
            if m:
                best = max(best, int(m.group(1)))
    return best


def submit(cfg, resume=False, test_only=False) -> int:
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    seeding.write_provenance(camp, cfg)

    start = resume_round(camp) if resume else 0
    if not os.path.exists(storage.policy_path(camp, start)):
        policy.init_policy_npz(storage.policy_path(camp, start), hidden=cfg.hidden,
                               logstd_init=cfg.logstd_init, seed=cfg.seed)
    # Compute the fixed shear-reward null ensemble once up front (no-op otherwise),
    # so rollout array tasks just load it. (run_rollout also computes it lazily
    # under a file lock if absent.)
    from jamrl import rollout
    rollout.ensure_null_ensemble(cfg, camp)

    dry = test_only or (shutil.which("sbatch") is None)
    payload = submit_round(cfg, camp, start, dry=dry, test_only=test_only)
    tag = " (dry-run)" if dry else ""
    print(f"[submit] campaign={camp} round={start}{tag}")
    print(f"[submit] rollout={payload['roll_jid']} learn={payload['learn_jid']} "
          f"post={payload['post_jid']} dep={payload['dependency']}")
    return 0


def maybe_resubmit(cfg, camp, r) -> None:
    """Step (d): launch round r+1 unless STOP exists or the campaign is done."""
    if os.path.exists(os.path.join(camp, "STOP")) or (r + 1) >= cfg.rounds:
        open(os.path.join(camp, "DONE"), "w").close()
        print(f"[learn] campaign complete at round {r} (DONE written).")
        return
    dry = shutil.which("sbatch") is None
    payload = submit_round(cfg, camp, r + 1, dry=dry)
    print(f"[learn] submitted round {r + 1}: {payload}")
