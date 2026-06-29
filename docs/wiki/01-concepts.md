# 01 — Concepts

> ⚙️ **Activate the `jamrl` conda env first** (`conda activate jamrl`) — every
> `jamrl` command in this wiki assumes it. See
> [06 — Environment & building](06-environment-and-building.md).

The mental model you need to not get lost. Read once, refer back as needed.

## What a campaign is

A **campaign** is one training run of the RL agent, identified by `--name` and
living under `--campaign-root`. One `jamrl submit` launches a **self-perpetuating
chain** of SLURM jobs: each round trains the policy a bit more and then submits
the next round itself. You do not babysit it — you submit once, and it runs until
it hits its round target or you stop it.

### The round chain

Each round `r` is three SLURM jobs:

```
policy(r) ──► ROLLOUT (array: W workers)  ──►  trajectories + jammed states
                          │ afterany (or afterok)
              ┌───────────┴───────────┐
              ▼                        ▼
         LEARN (1 job)           POSTPROCESS (array: shards)
         PPO/CEM update          diagonalize Hessians → spectra/moduli
         → policy(r+1)           → analysis/.../spectra_shard_*.npz
         → submits round r+1     (non-blocking; never stalls LEARN)
```

- **Rollout** (array job, one task per worker) runs episodes with the current
  policy and writes trajectories + jammed states.
- **Learn** (single job) aggregates all workers, runs the PPO/CEM update, writes
  `policy/round_{r+1}.npz`, appends a summary row, and **resubmits round r+1**
  (`maybe_resubmit` in [`src/jamrl/slurm.py`](../../src/jamrl/slurm.py)).
- **Postprocess** (array job, sharded) diagonalizes the saved jammed-state
  Hessians into vibrational spectra and moduli. It only *reads* saved states, so
  it never blocks learn or the next round.

Learn and postprocess both depend on the rollout array via SLURM's
`--dependency`, controlled by `--dependency-mode` (`afterany` = robust, runs even
if some rollout tasks failed; `afterok` = strict).

## Local vs cluster

- **`jamrl run-local`** runs the *whole* loop in a single process — all rounds,
  workers as a serial loop, no SLURM. Use it to smoke-test config and code.
- **`jamrl submit`** launches the SLURM chain above. Use it for real runs.

The learner has interchangeable backends via `--backend {auto,torch,numpy}`
(default `auto`). See the [backend section of the top-level README](../../README.md#learner-backend)
for the torch-vs-numpy tradeoffs — not duplicated here.

## Reward modes

The objective the agent optimizes is set by `--reward-mode` (default `density`).
Every mode measures the **final jammed state** at the campaign's fixed target
pressure `P` and rewards improvement over a zero-action **null protocol** at the
same `P`. The reward is computed inside the C++ `_core`.

| mode | terminal reward | baseline | weight (default) |
|---|---|---|---|
| `density` | `w_phi·(φ − φ_null)` — denser-than-null packings | `φ_null` | `--w-phi` (400) |
| `shear_modulus` | `w_G·(G − G_null)` — stiffer-in-shear packings | `G_null` | `--w-G` (200) |
| `speed` | `w_speed·(cost_null − cost)/cost_null` — fewer force evals | `cost_null` | `--w-speed` (200) |

Per-step cost (`--c-step`) and the failure/truncation penalties (`--fail-pen`,
`--trunc-pen`) are identical across modes; only the terminal term changes.
Baselines are computed once per `(N, P, seed)` and cached in `null_cache/`.

> **Rebuild after pulling reward changes.** Because the reward lives in C++, you
> must rebuild `_core` (`pip install -e .` or `scripts/build.sh`) after pulling
> any reward-mode change — on your laptop *and* on the cluster. See
> [Recipes ▸ FAQ](05-recipes.md#faq).

The objective each mode tracks in `status`/`plot`/notebooks is
`eval_dphi` (density), `eval_dG` (shear), or `eval_speed` (speed).

## How a campaign ends

Two ways, both via sentinel files at the campaign root (handled in
`maybe_resubmit`):

- **`DONE`** — written automatically by the last learn job when `r+1 >= rounds`.
- **`STOP`** — you create it (`touch <camp>/STOP`); the next learn job sees it,
  writes `DONE` instead of resubmitting, and the chain halts cleanly after the
  current round.

## Campaign directory layout

Created by `ensure_campaign_dirs` in
[`src/jamrl/storage.py`](../../src/jamrl/storage.py) — that file's path helpers
are the source of truth for everything below.

```
<campaign-root>/<name>/
├── config.yaml              # full resolved Config (reproducibility)
├── provenance.json          # git commit, config hash, timestamp
├── policy/
│   └── round_0000.npz       # actor policy per round (round_{r:04d}.npz)
├── checkpoints/
│   └── learner_round_0001.npz   # optimizer/CEM state for resuming
├── rollouts/
│   └── round_0000/
│       └── worker_000.npz   # per-worker trajectories (worker_{k:03d}.npz)
├── states/
│   └── round_0000/
│       └── worker_000.h5    # per-worker jammed-state Hessians
├── analysis/
│   ├── summary.parquet      # one row per round (append-only) — read by status/plot
│   ├── postprocess.parquet  # per-shard mechanics aggregate (append-only)
│   ├── round_0000/
│   │   └── spectra_shard_000.npz   # per-shard eigenvalues/moduli
│   ├── plots/summary.png    # written by `jamrl plot`
│   └── campaign_analysis.h5 # portable export written by `jamrl analyze`
├── rounds/
│   └── round_0000.json      # SLURM job ids: {roll_jid, learn_jid, post_jid, ...}
├── null_cache/
│   └── N1024_P1.000000e-03_s12345.txt   # null baselines, one tiny file per key
│                            #   (…__G.txt / …__cost.txt for shear/speed modes)
├── .sbatch/                 # rendered sbatch scripts (rollout/learn/postprocess_r####.sbatch)
└── logs/                    # job stdout/stderr (roll_r####_%A_%a.out, learn_r####_%j.out, …)
```

Key columns in `summary.parquet`: `round, episodes, mean_reward, eval_dphi,
eval_dG, eval_speed, eval_success, eval_cost_kevals, mean_absaP, mean_absaS,
mean_absgamma, Bbar, Gbar, dzbar, rattler_frac, shear_stable_frac, omega_star,
sigma_policy, wall_seconds, git_hash` (`SUMMARY_COLUMNS` in `storage.py`).

→ Next: [02 — CLI reference](02-cli-reference.md)
