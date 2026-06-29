# 05 — Recipes

> ⚙️ **Activate the `jamrl` conda env first** (`conda activate jamrl`) — every
> snippet below runs in that environment. See
> [06 — Environment & building](06-environment-and-building.md).

Copy-paste answers to "how do I…". Each links to the fuller page.

## Smoke-test locally

Run a tiny campaign in one process before burning cluster time:

```bash
conda activate jamrl
jamrl run-local --N 64 --rounds 3 --workers 2 --episodes-per-worker 2 \
                --T-cap 30 --name smoke
jamrl status --campaign campaigns/smoke
jamrl plot   --campaign campaigns/smoke
```

See [01 — Concepts ▸ local vs cluster](01-concepts.md#local-vs-cluster).

## Inspect SLURM wiring without queuing

```bash
jamrl submit --N 1024 --rounds 1000 --workers 64 --name big1k --test-only
```

Renders scripts into `campaigns/big1k/.sbatch/` and records the job graph; queues
nothing. See [03 ▸ dry-run](03-running-campaigns.md#dry-run-first).

## Train for shear stiffness

```bash
jamrl submit --N 256 --rounds 500 --reward-mode shear_modulus --w-G 200 \
             --partition cpu --name shear1
```

**Tune `--w-G`** so the training reward is `O(1)` (comparable to density mode's
~±1): run a handful of rounds, check `jamrl status`, and scale `--w-G` up or down
accordingly. `status`/`plot`/notebooks automatically track `eval_dG = ⟨G − G_null⟩`
in shear mode. See [01 ▸ reward modes](01-concepts.md#reward-modes).

## See how far a running campaign has gotten

```bash
jamrl status --campaign /home/data/$USER/campaigns/big1k
```

Works mid-campaign (reads the live parquet). See
[03 ▸ monitoring](03-running-campaigns.md#monitoring-a-running-campaign).

## Plot results while a campaign is still running

```bash
# quick on-cluster figure
jamrl plot --campaign <camp>

# full offline analysis (only completed rounds are included)
jamrl analyze --campaign <camp>
scp user@cluster:<camp>/analysis/campaign_analysis.h5 .
# open notebooks/01_training_curves.ipynb, edit the path cell
```

See [04 — Analysis](04-analysis.md).

## Stop a campaign now

```bash
touch /home/data/$USER/campaigns/big1k/STOP   # halts cleanly after the current round
```

## Resume a campaign later

```bash
jamrl submit --resume --name big1k --campaign-root /home/data/$USER/campaigns
```

(If you raised `--rounds`, delete a stale `DONE` sentinel first.) See
[03 ▸ resume](03-running-campaigns.md#resume-a-campaign).

## Drive everything from a YAML config

```bash
jamrl submit --config big1k.yaml --name big1k   # CLI flags still override the YAML
```

The resolved config is always saved to `<camp>/config.yaml`, so you can copy that
as a starting point.

## Re-evaluate a policy at larger N

```bash
jamrl eval --campaign <camp> --round 500 --N 4096
```

See [04 ▸ eval](04-analysis.md#re-evaluate-at-a-different-system-size).

## FAQ

**"OMP Error #15" when importing numpy/numba.** A second OpenMP runtime got
linked in. Build `_core` against the Python env's libomp (the default in
`scripts/build.sh`), and on macOS avoid linking Homebrew's libomp. See the
[OpenMP note in the top-level README](../../README.md#install--build).

**Reward looks wrong after pulling, or C++ changes aren't taking effect.** The
reward and physics live in the C++ `_core`, so any `cpp/` change needs a rebuild
(in the `jamrl` env): `conda activate jamrl && pip install -e . --no-build-isolation`
— on your laptop **and** on the cluster. Full instructions:
[06 — Environment & building](06-environment-and-building.md).

**Which objective column should I watch?** `eval_dphi` (density), `eval_dG`
(shear), or `eval_speed` (speed). `status`/`plot` pick the right one for your
`--reward-mode` automatically.

← Back to [wiki home](README.md)
