# 02 — CLI reference

> ⚙️ **Activate the `jamrl` conda env first** (`conda activate jamrl`) — every
> `jamrl` invocation below runs in that environment. See
> [06 — Environment & building](06-environment-and-building.md).

Every `jamrl` subcommand and every config flag. Commands come from
`build_parser` in [`src/jamrl/cli.py`](../../src/jamrl/cli.py); flags are
auto-generated from the `Config` dataclass in
[`src/jamrl/config.py`](../../src/jamrl/config.py).

```
jamrl <command> [options]
```

## Commands

Commands tagged **(SLURM-internal)** are run for you by the generated sbatch
scripts — you normally never call them by hand.

### `submit` — launch a cluster campaign
Submits the self-perpetuating SLURM chain (rollout → learn → postprocess; learn
resubmits the next round). Accepts all config flags below.

| flag | meaning |
|---|---|
| `--resume` | start from the highest existing `policy/round_*.npz` instead of round 0 |
| `--test-only` | render scripts and validate SLURM directives without queuing |

```bash
jamrl submit --N 1024 --rounds 1000 --workers 64 --partition cpu --name big1k
```

### `run-local` — run the whole loop in one process
No SLURM. Iterates all `--rounds` in a serial loop; exits early if a `STOP`
sentinel appears. Accepts all config flags. Best for smoke tests.

```bash
jamrl run-local --N 64 --rounds 3 --workers 2 --episodes-per-worker 2 --name smoke
```

### `status` — print campaign progress
Reads `analysis/summary.parquet` (works mid-campaign). Shows rounds completed,
DONE/STOP state, a tail of recent rounds (reward, objective, success, sigma), the
last eval's mechanics, and the last round's job ids.

| flag | required | meaning |
|---|---|---|
| `--campaign` | yes | path to the campaign directory |

### `plot` — regenerate summary figures
Writes `analysis/plots/summary.png` from the parquet. Works mid-campaign.

| flag | required | meaning |
|---|---|---|
| `--campaign` | yes | path to the campaign directory |

### `analyze` — export a portable analysis HDF5
Condenses the campaign into `analysis/campaign_analysis.h5` for offline
notebooks. Reads only completed rounds, so it works **mid-campaign**.

| flag | default | meaning |
|---|---|---|
| `--campaign` | (required) | path to the campaign directory |
| `--spectra-stride` | 10 | sample VDOS/mechanics every N rounds |
| `--traj-stride` | 25 | sample trajectory/observation/action data every N rounds |
| `--out` | `<camp>/analysis/campaign_analysis.h5` | output path |

### `eval` — greedy evaluation of a saved policy
Runs a deterministic rollout of one round's policy; optionally at a different
system size `--N` (useful to transfer a trained policy to larger packings).

| flag | default | meaning |
|---|---|---|
| `--campaign` | (required) | path to the campaign directory |
| `--round` | (required) | which round's policy to evaluate |
| `--N` | 0 | override system size (0 = use the campaign's N) |

### `compact` — shrink stored Hessian data
Down-converts old per-state Hessians to compact spectra to reclaim disk.

| flag | default | meaning |
|---|---|---|
| `--campaign` | (required) | path to the campaign directory |
| `--round-range` | `""` (all) | `lo:hi` range of rounds to compact |
| `--keep-spectrum` | off | keep the spectral data after compacting |

### `rollout` — one rollout worker *(SLURM-internal)*
Runs episodes for a single worker of one round.

| flag | required | meaning |
|---|---|---|
| `--campaign` / `--round` / `--worker` | yes | campaign, round, worker index |

### `learn` — policy update + resubmit *(SLURM-internal)*
Aggregates the round's rollouts, updates the policy, evaluates, appends the
summary row, and resubmits the next round.

| flag | default | meaning |
|---|---|---|
| `--campaign` / `--round` | (required) | campaign, round |
| `--no-resubmit` | off | do the update but do **not** submit the next round |

### `postprocess` — diagonalize Hessians *(SLURM-internal)*
Computes spectra/moduli for one shard of a round's jammed states.

| flag | default | meaning |
|---|---|---|
| `--campaign` / `--round` | (required) | campaign, round |
| `--shard` / `--nshards` | 0 / 1 | this shard index and the total shard count |

## Config flags

Every `Config` field becomes a flag: `field_name` → `--field-name`. Resolution
order is **dataclass defaults < `--config YAML` < explicit CLI flags** (CLI
wins). Use `--config run.yaml` to load a YAML file (any flag still overrides it).
`--hidden` and `--eval-seeds` are **comma-separated** (e.g. `--hidden 64,64`).

### System / environment
| flag | type | default | meaning |
|---|---|---|---|
| `--N` | int | 1024 | number of particles |
| `--P` | float | 1e-3 | target pressure |
| `--phi0` | float | 0.80 | initial packing fraction |

### Agent loading / relaxation
| flag | type | default | meaning |
|---|---|---|---|
| `--kappa-P` | float | 1.0 | pressure-loading scale |
| `--kappa-sigma` | float | 0.5 | shear-loading scale |
| `--n-relax` | int | 20 | relaxation steps per env step |
| `--T-cap` | int | 60 | episode length cap |

### Reward
| flag | type | default | meaning |
|---|---|---|---|
| `--reward-mode` | str | density | `density` \| `shear_modulus` \| `speed` |
| `--w-phi` | float | 400.0 | density-mode weight on `φ − φ_null` |
| `--w-G` | float | 200.0 | shear-mode weight on `G − G_null` (tune so reward ~O(1)) |
| `--w-speed` | float | 200.0 | speed-mode weight on `(cost_null − cost)/cost_null` |
| `--c-step` | float | 0.01 | per-step cost |
| `--fail-pen` | float | 2.0 | failure (blowup) penalty |
| `--trunc-pen` | float | 0.5 | truncation penalty |
| `--quiesce-tol` | float | 0.05 | quiescence tolerance for early finish |
| `--quiesce-n` | int | 3 | consecutive quiescent steps to finish |
| `--finish-cap` | int | 12000 | lower bound on finish-and-measure iterations |
| `--finish-cap-max` | int | 60000 | hard ceiling (raise to reach jamming at very low P) |

### Numerical tolerances
| flag | type | default | meaning |
|---|---|---|---|
| `--ftol-abs` | float | 1e-10 | absolute force tolerance |
| `--ftol-rel-P` | float | 1e-5 | pressure-relative force tolerance |
| `--ptol` | float | 1e-4 | pressure tolerance |

### Learner
| flag | type | default | meaning |
|---|---|---|---|
| `--algo` | str | ppo | `ppo` \| `cem` |
| `--backend` | str | auto | PPO backend: `auto` \| `torch` \| `numpy` |
| `--lr` | float | 3e-4 | learning rate |
| `--gamma` | float | 0.995 | discount factor |
| `--lam` | float | 0.95 | GAE lambda |
| `--clip` | float | 0.2 | PPO clip ratio |
| `--ppo-epochs` | int | 6 | epochs per round |
| `--minibatch` | int | 1024 | minibatch size |
| `--ent-coef` | float | 3e-3 | entropy bonus |
| `--vf-coef` | float | 0.5 | value-loss coefficient |
| `--logstd-init` | float | -0.5 | initial policy log-std |
| `--hidden` | tuple | 64,64 | hidden layer sizes (comma-separated) |
| `--cem-pop` | int | 64 | CEM population size |
| `--cem-elite-frac` | float | 0.25 | CEM elite fraction |
| `--cem-sigma0` | float | 0.3 | CEM initial sigma |
| `--cem-eps-per-cand` | int | 4 | CEM episodes per candidate |
| `--device` | str | cpu | `cpu` \| `cuda` |

### Campaign / parallelism
| flag | type | default | meaning |
|---|---|---|---|
| `--rounds` | int | 1000 | training rounds (campaign length) |
| `--workers` | int | 64 | parallel rollout workers (array size) |
| `--episodes-per-worker` | int | 8 | episodes each worker collects per round |
| `--eval-seeds` | tuple | 101,102,103,104,105,106 | fixed greedy-eval seeds (comma-separated) |
| `--parallel-mode` | str | episode | `episode` \| `intra` |
| `--threads-per-task` | int | 16 | CPU threads per task |

### Data / checkpointing
| flag | type | default | meaning |
|---|---|---|---|
| `--save-hessian` | str | sparse | `none` \| `spectrum` \| `sparse` \| `dense` |
| `--hessian-stride` | int | 1 | save the Hessian every N rounds |
| `--compression` | str | gzip | HDF5 compression codec |

### Postprocess (box-VDOS + projections)
| flag | type | default | meaning |
|---|---|---|---|
| `--dos-full` | bool | false | also diagonalize the full enthalpy Hessian (box DOF included) |
| `--proj-k` | int | 60 | # of lowest relaxation modes for box-VDOS + projection |

### Node-local scratch (HPC)
| flag | type | default | meaning |
|---|---|---|---|
| `--node-scratch` | str | `""` (off) | stage heavy outputs here, copy to campaign at task end; supports `$VARS` (e.g. `$TMPDIR`). Also via `JAMRL_NODE_SCRATCH` (overrides config). |

### SLURM
| flag | type | default | meaning |
|---|---|---|---|
| `--partition` | str | `""` | SLURM partition |
| `--account` | str | `""` | SLURM account |
| `--time-rollout` | str | 02:00:00 | rollout walltime |
| `--time-learn` | str | 00:20:00 | learn walltime |
| `--time-post` | str | 01:00:00 | postprocess walltime |
| `--mem-rollout` | str | 8G | rollout memory |
| `--mem-learn` | str | 8G | learn memory |
| `--mem-post` | str | 16G | postprocess memory |
| `--dependency-mode` | str | afterany | `afterany` (robust) \| `afterok` (strict) |
| `--min-worker-frac` | float | 0.6 | min fraction of workers that must succeed to proceed |

### Misc
| flag | type | default | meaning |
|---|---|---|---|
| `--seed` | int | 12345 | base random seed |
| `--campaign-root` | str | ./campaigns | root directory for campaigns |
| `--name` | str | run | campaign name (directory under the root) |
| `--config` | str | — | YAML config to load (CLI flags still override) |

→ Next: [03 — Running campaigns](03-running-campaigns.md)
