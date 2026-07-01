"""jamrl command-line interface (plan section 5.3)."""
from __future__ import annotations

import argparse
import os
import sys

from jamrl import config, policy, seeding, storage
from jamrl.config import Config


def _add_common(p):
    p.add_argument("--config", help="YAML config file (CLI flags override it)")
    config.add_arguments(p)


def _resolve(args) -> Config:
    return config.from_args(args)


def _init_campaign(cfg: Config) -> str:
    camp = storage.campaign_dir(cfg)
    storage.ensure_campaign_dirs(camp)
    cfg.save_yaml(os.path.join(camp, "config.yaml"))
    seeding.write_provenance(camp, cfg)
    p0 = storage.policy_path(camp, 0)
    if not os.path.exists(p0):
        policy.init_policy_npz(p0, hidden=cfg.hidden, act_dim=config.act_dim(cfg),
                               logstd_init=cfg.logstd_init, seed=cfg.seed)
    # Fixed shear-reward baseline: a per-campaign null ensemble (no-op otherwise).
    from jamrl import rollout
    ens = rollout.ensure_null_ensemble(cfg, camp)
    if ens is not None:
        print(f"[init] null ensemble: {ens['n_jammed']}/{ens['n_null']} jammed, "
              f"G_mean={ens['G_mean']:.5f} (median {ens['G_median']:.5f})")
    return camp


# ----------------------------------------------------------------------- #
def cmd_run_local(args):
    from jamrl import learn, rollout

    cfg = _resolve(args)
    camp = _init_campaign(cfg)
    print(f"[run-local] campaign={camp} algo={cfg.algo} N={cfg.N} "
          f"rounds={cfg.rounds} workers={cfg.workers} eps/worker={cfg.episodes_per_worker}")
    for r in range(cfg.rounds):
        if os.path.exists(os.path.join(camp, "STOP")):
            print("[run-local] STOP sentinel found; halting.")
            break
        if cfg.algo != "cem":
            for k in range(cfg.workers):
                rollout.run_rollout(cfg, camp, r, k)
        else:
            # CEM evaluates candidates inside the learner; a single dummy rollout
            # provides the aggregate bookkeeping the learner expects.
            rollout.run_rollout(cfg, camp, r, 0)
        stats = learn.learn_round(cfg, camp, r)
        # Show the objective tracked for this reward mode (dphi | dG | speed).
        obj_key = {"shear_modulus": "eval_dG", "speed": "eval_speed"}.get(
            cfg.reward_mode, "eval_dphi")
        print(f"  round {r:4d}: mean_reward={stats.get('mean_reward', float('nan')):+.3f} "
              f"{obj_key}={stats.get(obj_key, float('nan')):+.4f} "
              f"success={stats.get('eval_success', float('nan')):.2f} "
              f"sigma={stats.get('sigma_policy', float('nan')):.3f}")
    else:
        open(os.path.join(camp, "DONE"), "w").close()
    return 0


def cmd_rollout(args):
    from jamrl import rollout

    cfg = Config.from_yaml(os.path.join(args.campaign, "config.yaml"))
    rollout.run_rollout(cfg, args.campaign, args.round, args.worker)
    return 0


def cmd_learn(args):
    from jamrl import learn, slurm

    cfg = Config.from_yaml(os.path.join(args.campaign, "config.yaml"))
    stats = learn.learn_round(cfg, args.campaign, args.round)
    print(f"[learn] round {args.round}: {stats}")
    # step (d): self-resubmit the next round unless STOP / done
    if not args.no_resubmit:
        slurm.maybe_resubmit(cfg, args.campaign, args.round)
    return 0


def cmd_postprocess(args):
    from jamrl import postprocess

    cfg = Config.from_yaml(os.path.join(args.campaign, "config.yaml"))
    postprocess.run_postprocess(cfg, args.campaign, args.round, args.shard, args.nshards)
    return 0


def cmd_submit(args):
    from jamrl import slurm

    cfg = _resolve(args)
    return slurm.submit(cfg, resume=args.resume, test_only=args.test_only)


def cmd_status(args):
    from jamrl import analyze

    analyze.print_status(args.campaign)
    return 0


def cmd_plot(args):
    from jamrl import analyze

    analyze.plot_campaign(args.campaign)
    return 0


def cmd_compact(args):
    from jamrl import postprocess

    a, b = (args.round_range.split(":") + ["", ""])[:2] if args.round_range else ("", "")
    lo = int(a) if a else 0
    hi = int(b) if b else None
    postprocess.compact(args.campaign, lo, hi, keep_spectrum=args.keep_spectrum)
    return 0


def cmd_analyze(args):
    from jamrl import campaign_analysis

    out = campaign_analysis.build_campaign_analysis(
        args.campaign,
        spectra_stride=args.spectra_stride,
        traj_stride=args.traj_stride,
        out_path=args.out,
    )
    print(f"[analyze] wrote {out}")
    return 0


def cmd_eval(args):
    from jamrl import learn

    cfg = Config.from_yaml(os.path.join(args.campaign, "config.yaml"))
    if args.N:
        cfg = cfg.replace(N=args.N)
    stats = learn.greedy_eval(cfg, args.campaign, args.round)
    print(f"[eval] round {args.round} N={cfg.N}: {stats}")
    return 0


# ----------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jamrl", description="Box-control jamming RL")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("submit", help="submit a self-perpetuating SLURM campaign")
    _add_common(ps)
    ps.add_argument("--resume", action="store_true")
    ps.add_argument("--test-only", action="store_true")
    ps.set_defaults(func=cmd_submit)

    pr = sub.add_parser("run-local", help="run the whole loop in one process (no SLURM)")
    _add_common(pr)
    pr.set_defaults(func=cmd_run_local)

    pro = sub.add_parser("rollout", help="one rollout array task")
    pro.add_argument("--campaign", required=True)
    pro.add_argument("--round", type=int, required=True)
    pro.add_argument("--worker", type=int, required=True)
    pro.set_defaults(func=cmd_rollout)

    pl = sub.add_parser("learn", help="aggregate -> update -> write -> resubmit")
    pl.add_argument("--campaign", required=True)
    pl.add_argument("--round", type=int, required=True)
    pl.add_argument("--no-resubmit", action="store_true")
    pl.set_defaults(func=cmd_learn)

    pp = sub.add_parser("postprocess", help="diagonalize Hessians -> spectra/moduli")
    pp.add_argument("--campaign", required=True)
    pp.add_argument("--round", type=int, required=True)
    pp.add_argument("--shard", type=int, default=0)
    pp.add_argument("--nshards", type=int, default=1)
    pp.set_defaults(func=cmd_postprocess)

    pst = sub.add_parser("status", help="campaign progress + reward tail")
    pst.add_argument("--campaign", required=True)
    pst.set_defaults(func=cmd_status)

    ppl = sub.add_parser("plot", help="regenerate figures from parquet")
    ppl.add_argument("--campaign", required=True)
    ppl.set_defaults(func=cmd_plot)

    pc = sub.add_parser("compact", help="down-convert old Hessians to spectra")
    pc.add_argument("--campaign", required=True)
    pc.add_argument("--keep-spectrum", action="store_true")
    pc.add_argument("--round-range", default="")
    pc.set_defaults(func=cmd_compact)

    pe = sub.add_parser("eval", help="greedy eval of a saved policy (optionally at new N)")
    pe.add_argument("--campaign", required=True)
    pe.add_argument("--round", type=int, required=True)
    pe.add_argument("--N", type=int, default=0)
    pe.set_defaults(func=cmd_eval)

    pan = sub.add_parser("analyze", help="condense campaign into portable h5 for offline notebooks")
    pan.add_argument("--campaign", required=True)
    pan.add_argument("--spectra-stride", type=int, default=10,
                     help="sample VDOS/mechanics every N rounds (default: 10)")
    pan.add_argument("--traj-stride", type=int, default=25,
                     help="sample trajectory/obs/action data every N rounds (default: 25)")
    pan.add_argument("--out", default=None,
                     help="output path (default: <campaign>/analysis/campaign_analysis.h5)")
    pan.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
