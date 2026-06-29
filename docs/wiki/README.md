# jamrl wiki

A task-oriented reference for **driving the `jamrl` binary** through the
box-control jamming RL workflow. When you lose the thread of "which command does
X again, and what are its flags?", start here.

This wiki is intentionally practical: it tells you *how to do things* and where
every flag and file lives. For the *physics* and *internals* see the deeper docs
linked at the bottom.

> ⚙️ **Before any `jamrl` command, activate the `jamrl` conda environment:**
> `conda activate jamrl`. It is the only supported environment for this repo —
> use it for the CLI, building `_core`, tests, and scripts, every time. See
> [06 — Environment & building `_core`](06-environment-and-building.md).

## I want to… → go here

| Goal | Page / section |
|---|---|
| Set up the env / (re)build `_core` after code changes | [06 — Environment & building](06-environment-and-building.md) |
| Understand the big picture (campaign, rounds, reward modes, files) | [01 — Concepts](01-concepts.md) |
| Look up a command or a config flag | [02 — CLI reference](02-cli-reference.md) |
| Smoke-test locally before touching the cluster | [05 — Recipes ▸ smoke-test](05-recipes.md#smoke-test-locally) |
| Launch a full cluster campaign | [03 — Running campaigns ▸ launch](03-running-campaigns.md#launch-a-campaign) |
| Dry-run / inspect SLURM wiring without queuing | [03 ▸ dry-run](03-running-campaigns.md#dry-run-first) |
| Check how far a running campaign has gotten | [03 ▸ monitoring](03-running-campaigns.md#monitoring-a-running-campaign) |
| Stop a campaign cleanly / resume it later | [03 ▸ stop](03-running-campaigns.md#stop-a-campaign-cleanly) · [03 ▸ resume](03-running-campaigns.md#resume-a-campaign) |
| Train for shear stiffness instead of density | [05 ▸ shear mode](05-recipes.md#train-for-shear-stiffness) |
| Pull results to my laptop and plot them | [04 — Analysis](04-analysis.md) |
| Run analysis **mid-campaign** | [04 ▸ analyze](04-analysis.md#jamrl-analyze) |
| Re-evaluate a trained policy at a larger N | [04 ▸ eval](04-analysis.md#re-evaluate-at-a-different-system-size) |
| Fix "OMP Error #15" / "reward looks wrong" | [05 — Recipes ▸ FAQ](05-recipes.md#faq) |

## Pages

1. [Concepts](01-concepts.md) — mental model: campaigns, the round chain, reward
   modes, sentinels, the campaign directory layout.
2. [CLI reference](02-cli-reference.md) — every subcommand and every config flag.
3. [Running campaigns](03-running-campaigns.md) — submit, dry-run, resume, stop,
   node-scratch staging, monitoring.
4. [Analysis](04-analysis.md) — `analyze` → scp → notebooks; `eval`; `compact`.
5. [Recipes](05-recipes.md) — copy-paste "how do I…" cookbook + FAQ.
6. [Environment & building `_core`](06-environment-and-building.md) — activate the
   `jamrl` env, build/rebuild the C++ core.

## Deeper docs (not duplicated here)

- [`../../README.md`](../../README.md) — install/build, backends, top-level overview.
- [`../physics.md`](../physics.md) — state vector, pair kernel, moduli/Hessian
  formulas, numerical-hazard fixes.
- [`../plans/jamrl_hpc_implementation_plan.md`](../plans/jamrl_hpc_implementation_plan.md)
  — the full implementation spec (architecture, storage schema, phases).
