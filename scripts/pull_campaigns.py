#!/usr/bin/env python3
"""Pull campaign analysis artifacts from the HPC cluster into the local campaigns/ dir.

By default this transfers only the lightweight analysis outputs (the portable
`campaign_analysis.h5`, the summary/postprocess parquets, and the plots) — NOT the
heavy raw rollouts/states, which can be hundreds of GB. Use --full to mirror an
entire campaign directory.

Examples
--------
    # pull analysis artifacts for every campaign on the cluster
    python scripts/pull_campaigns.py --host treado@cluster.address

    # pull just two campaigns
    python scripts/pull_campaigns.py --host treado@cluster.address \
        --name big_first_test_v3 --name short_test

    # set the host once via env var
    export JAMRL_CLUSTER=treado@cluster.address
    python scripts/pull_campaigns.py

    # mirror the FULL campaign (heavy — raw rollouts + states included)
    python scripts/pull_campaigns.py --name big_first_test_v3 --full
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

DEFAULT_REMOTE_ROOT = "/home/treado/data/jamrl/campaigns"

# analysis artifacts worth pulling (relative to a campaign dir). Globs allowed.
ARTIFACTS = [
    "config.yaml",
    "analysis/campaign_analysis.h5",
    "analysis/summary.parquet",
    "analysis/postprocess.parquet",
    "analysis/plots",  # directory
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def ssh_list_campaigns(host: str, remote_root: str) -> list[str]:
    """Return campaign names (dir basenames) present under remote_root."""
    cmd = ["ssh", host, f"ls -1 {shlex.quote(remote_root)}"]
    res = run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"[pull] failed to list remote campaigns:\n{res.stderr}")
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def scp_artifact(host: str, remote_root: str, name: str, rel: str, local_root: str) -> None:
    """scp one artifact (file or dir) for campaign `name`; skip if it doesn't exist."""
    remote = f"{remote_root}/{name}/{rel}"
    local = os.path.join(local_root, name, rel)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    # -r handles directories; harmless for single files. -p preserves mtimes.
    cmd = ["scp", "-rp", f"{host}:{remote}", local]
    res = run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # missing artifact (e.g. postprocess.parquet absent) is not fatal
        msg = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "no such file"
        print(f"    skip {rel}: {msg}")


def scp_full(host: str, remote_root: str, name: str, local_root: str) -> None:
    remote = f"{remote_root}/{name}"
    local = os.path.join(local_root, name)
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    cmd = ["scp", "-rp", f"{host}:{remote}", local]
    res = run(cmd)
    if res.returncode != 0:
        print(f"    WARNING: full transfer of {name} returned {res.returncode}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("JAMRL_CLUSTER"),
                    help="ssh target, e.g. treado@cluster.address "
                         "(or set $JAMRL_CLUSTER)")
    ap.add_argument("--remote-root", default=os.environ.get("JAMRL_REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
                    help=f"remote campaigns dir (default: {DEFAULT_REMOTE_ROOT})")
    ap.add_argument("--local-root", default="campaigns",
                    help="local destination dir (default: ./campaigns)")
    ap.add_argument("--name", action="append", default=[],
                    help="campaign name to pull (repeatable); default: all")
    ap.add_argument("--full", action="store_true",
                    help="mirror the ENTIRE campaign dir (heavy: raw rollouts + states)")
    ap.add_argument("--list", action="store_true",
                    help="just list remote campaigns and exit")
    args = ap.parse_args(argv)

    if not args.host:
        sys.exit("[pull] no --host given and $JAMRL_CLUSTER is unset. "
                 "Pass --host treado@cluster.address")

    if args.list:
        for c in ssh_list_campaigns(args.host, args.remote_root):
            print(c)
        return 0

    names = args.name or ssh_list_campaigns(args.host, args.remote_root)
    if not names:
        print("[pull] no campaigns found.")
        return 0

    print(f"[pull] host={args.host}")
    print(f"[pull] remote={args.remote_root}  ->  local={os.path.abspath(args.local_root)}")
    print(f"[pull] {'FULL mirror' if args.full else 'analysis artifacts'} for "
          f"{len(names)} campaign(s): {', '.join(names)}\n")

    for name in names:
        print(f"[pull] {name}")
        if args.full:
            scp_full(args.host, args.remote_root, name, args.local_root)
        else:
            for rel in ARTIFACTS:
                scp_artifact(args.host, args.remote_root, name, rel, args.local_root)
        print()

    print(f"[pull] done -> {os.path.abspath(args.local_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
