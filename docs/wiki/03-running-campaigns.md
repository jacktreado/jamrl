# 03 — Running campaigns

> ⚙️ **Activate the `jamrl` conda env first** (`conda activate jamrl`) — run
> every `jamrl` command here in that environment, including on the cluster. See
> [06 — Environment & building](06-environment-and-building.md).

Operating a real cluster campaign end to end. For the concepts behind the round
chain and sentinels, see [01 — Concepts](01-concepts.md).

## Launch a campaign

One `jamrl submit` starts the whole self-perpetuating chain:

```bash
jamrl submit --N 1024 --rounds 1000 --workers 64 --episodes-per-worker 8 \
             --threads-per-task 16 --partition cpu --account myproj \
             --save-hessian spectrum \
             --campaign-root /home/data/$USER/campaigns \
             --node-scratch '$TMPDIR' \
             --name big1k
```

Point `--campaign-root` at **durable shared storage** — it is the campaign's
source of truth and the only thing that crosses node boundaries. `submit` creates
the directory, writes `config.yaml` + `provenance.json`, initializes
`policy/round_0000.npz`, and queues round 0's three jobs (`submit` /
`submit_round` in [`src/jamrl/slurm.py`](../../src/jamrl/slurm.py)).

A YAML config can replace flags; CLI flags still override it:

```bash
jamrl submit --config big1k.yaml --name big1k
```

## Dry-run first

Validate the wiring without queuing anything:

```bash
jamrl submit --N 1024 --rounds 1000 --workers 64 --name big1k --test-only
```

`--test-only` renders the sbatch scripts into `<camp>/.sbatch/` and records the
job graph in `rounds/round_0000.json`, using `sbatch --test-only` to validate
directives if `sbatch` is present. Dry mode also engages automatically on any
host where `sbatch` is missing (e.g. your laptop), so you can inspect the
generated scripts anywhere.

## Monitoring a running campaign

Both commands read the live `analysis/summary.parquet`, so they work while the
campaign is still going:

```bash
jamrl status --campaign /home/data/$USER/campaigns/big1k
```

Shows rounds completed, `DONE`/`STOP` state, a tail of recent rounds (reward,
objective, success, sigma), the last eval's mechanics, and the most recent
round's SLURM job ids.

```bash
jamrl plot --campaign /home/data/$USER/campaigns/big1k
```

Writes `analysis/plots/summary.png` (training reward, the active objective,
success rate, G, Δz, |a_P|). For richer offline plots, see
[04 — Analysis](04-analysis.md).

You can also tail the SLURM logs under `<camp>/logs/` (`roll_r####_%A_%a.out`,
`learn_r####_%j.out`, `post_r####_%A_%a.out`).

## Stop a campaign cleanly

Create a `STOP` sentinel at the campaign root:

```bash
touch /home/data/$USER/campaigns/big1k/STOP
```

The chain does **not** die mid-round. The next learn job sees `STOP`, writes a
`DONE` sentinel instead of resubmitting, and the chain halts after the current
round finishes (`maybe_resubmit` in `slurm.py`). The auto-`DONE` at the
`--rounds` limit works the same way.

## Resume a campaign

To continue after a pause, preemption, or the round target being raised:

```bash
jamrl submit --resume --name big1k --campaign-root /home/data/$USER/campaigns
```

`--resume` scans `policy/` for the highest `round_*.npz` and restarts from there
(`resume_round` in `slurm.py`). The learn job picks up the matching
`checkpoints/learner_round_*.npz` (optimizer/CEM state) if present, otherwise
loads the policy npz directly. (If you raised `--rounds`, remove a stale `DONE`
sentinel first or it will immediately re-finish.)

## Node-scratch staging

On clusters where the shared filesystem is slow for many small concurrent
writes, stage each task's **heavy outputs** (trajectory npz, jammed-state h5,
spectra npz) to the node's local disk and copy them to the persistent campaign
when the task finishes:

```bash
jamrl submit ... --node-scratch '$TMPDIR'
```

- Set it via `--node-scratch` or the `JAMRL_NODE_SCRATCH` env var (the env var
  overrides the config, so an admin can set a cluster default). Supports `$VARS`
  and `~`; an unset/unwritable path silently falls back to writing in place.
- **Reads** always come from the persistent campaign, and genuinely shared files
  (the `null_cache/` shards, the summary parquets) stay there too — a per-node
  scratch copy could not be shared across the nodes of a distributed campaign.
- Staging is transparent: unset (the default) writes happen in place, so
  `run-local` and laptop runs are unaffected.

Source: [`src/jamrl/staging.py`](../../src/jamrl/staging.py). See also the
top-level README's
[HPC storage section](../../README.md#hpc-storage-persistent-campaign--node-local-scratch).

→ Next: [04 — Analysis](04-analysis.md)
