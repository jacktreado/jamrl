# Agent action / reward expansion — VDOS-directed moves (and swaps later)

Status: planned (2026-07-01). Owner: Jack. Context below, then the phased plan.

## Why

Shear campaigns (e.g. `sm_N256_P1e-5`) plateau: with a healthy exploration band
(policy σ ≈ 0.6, neither collapsed nor runaway) the eval `G − G_null` stays flat
at ~5e-4 over 100+ rounds. Diagnosis from the capped run's analysis h5:

- mean final G ≈ 0.0051
- config-to-config spread std(G) ≈ 0.0033 (**~65% of the mean**)
- mean improvement (G − G_null) ≈ 0.00047 (~9% over null)
- improvement / spread ≈ **0.14 σ**

So the agent's ~9% mean lift is *real* but buried under enormous initial-config
variance. Two separate problems: (1) the **training reward SNR** is swamped by
config variance; (2) the **box-control action space** (aP, aS only) is a weak
lever on G — ~9% is near its ceiling. Fix both.

## Confirmed code facts (map, 2026-07-01)

- Action space: `ACT_DIM = 2` (aP pressure, aS shear). Defined `policy.py:14` /
  `cpp/include/jamcore/env.hpp:29`. `Env::step(double aP, double aS)` (`env.cpp:153`).
- Macro-step: clamp actions → loads (`Peff`, `sigF`) → `lbfgs_relax` (`env.cpp:165`)
  → evaluate → `measure_obs_extras` (G + VDOS) → converge/finish-and-measure.
- Reward (`env.cpp:37-49`, terminal at jamming): `w_G*(G/G_null - 1)`; dense
  `-c_step`/step; `-fail_pen`, `-trunc_pen`. Reward **G_null is currently the
  broadcast ensemble mean**; training is NOT paired. (Eval already IS per-seed
  paired — that's why eval_dG is the cleaner metric.)
- Per-seed null baselines already implemented: `ensure_null_baselines(cfg,camp,seeds)`
  (`rollout.py:40`) → `_core.compute_null_baselines`; C++ batch runner already
  indexes per-seed `g_null`. `run_rollout` shear branch (`rollout.py:175-181`) just
  hardcodes the broadcast mean instead of calling it.
- VDOS: env diagonalizes the full-enthalpy Hessian (2N+2, box DOF incl.) each step
  in `measure_obs_extras` (`env.cpp:91-121`) via `eigvals_full_spectrum` (values
  only). Eigenvectors available via `eigvecs_full(sys,k)` (`hessian.hpp:39`), used
  today only in postprocess. Obs `w1..w5` (`obs[11..15]`) are the 5 lowest real
  eigenfrequencies. Marginal cost to retain eigenvectors + one matvec/step ≈ 1-5%.
- Obs: `OBS_DIM = 16` = 10 base + G(`obs[10]`) + 5 VDOS. Normalized by RunningNorm.

## Plan (do in order)

### (a) Confirm the ceiling — DONE (proxy). Optional exact check.
Proxy from analysis h5 already shows improvement ≈ 0.14σ of the G-spread.
Exact number on the cluster: read `null_ensemble.json` → `G_std`, compare to
`summary.eval_dG`. If eval_dG ≲ G_std, box control is exhausted → need (c).

### (b) Paired (common-random-numbers) reward baseline — IMPLEMENTED, but does NOT help.
Hypothesis was: pairing each episode against its own seed's null cancels the ~65%
config variance. **Empirically false for this system (verified N=32, 2026-07-01).**
- Implemented + tested, gated by `paired_null_G: bool = False` (default off, no C++
  change): `run_rollout` shear branch now calls `ensure_null_baselines(cfg,camp,seeds)`
  when the flag is on. Per-seed null G flows to reward AND obs normalization.
- Validation finding (the important part): the null baseline is CORRECT — a true
  zero-action episode reproduces `compute_null_baselines` G exactly (corr=1.000).
  BUT the jamming quench is **path-chaotic**: any finite action jumps the system to a
  different basin, so corr(G_agent, G_null_seed) is only ~0.3 (random policy) rising
  to ~0.47 as actions→0. Pairing subtracts a weakly-correlated reference, so
  `std(G_agent − G_null) > std(G_agent)` at every action scale tested — pairing
  *increases* variance. The 65% spread is basin/path variance, not initial-config
  variance; CRN can only cancel ~corr² ≈ 9%.
- CONCLUSION: keep the code (default off; it's the honest per-seed baseline the eval
  already uses, may help a well-converged small-action policy marginally) but DO NOT
  expect it to fix the plateau; do not prioritize a paired campaign. Real SNR levers:
  the PPO value-function baseline (verify it predicts outcome from early obs) and,
  decisively, the (c) action lever. Caveat: measured at N=32; variance ~1/N so N=256
  spread is smaller and corr *may* be higher — but path-chaos is scale-general, so
  don't bet on it. Definitive test would be a paired-vs-unpaired N=256 run; better to
  spend that compute on (c).

### (c) VDOS-directed moves DURING the protocol — IMPLEMENTED (2026-07-01).
Config `k_vdos_moves` (0=off) + `vdos_move_amp`. C++: `EnvConfig.{k_vdos_moves,
vdos_move_amp}`; `Env::vdos_vecs` retained from `eigvecs_full` in `measure_obs_extras`
when moves on (else eigenvalues-only, byte-identical); `Env::step(const VectorXd&)`
applies `sys.x.head(2N) += amp*c_j*ê_j` before `lbfgs_relax` (2-arg overload kept for
tests/null); `rollout.cpp` passes full action + sizes `out.act` to `pol.act_dim()`;
pybind exposes the fields + a vector `step`. Python: `config.act_dim(cfg)=2+k_vdos_moves`
threaded into every policy-construction site (cli/learn/slurm/torch_backend/cem).
Tests: `tests/test_vdos_moves.py` (plumbing, action widened to 7, kick changes dynamics).
Verified: run-local shear campaign trains a 7-dim policy end-to-end; all 110 non-slow
tests pass. Rebuild via `/opt/anaconda3/envs/jamrl/bin/pip install -e . --no-build-isolation`.
Run: `jamrl submit --config ... --reward-mode shear_modulus --k-vdos-moves 5 --vdos-move-amp 0.02 --name <fresh>`.
Design details (unchanged from below):
Agent outputs k extra continuous coefficients `c_j`; each macro-step, BEFORE the
held-load relaxation, displace along the current soft modes:
`sys.x += amp * Σ_j c_j * ê_j` (ê_j = unit particle-DOF part of the j-th lowest
real mode already computed for the obs). The existing `lbfgs_relax` then settles
the perturbed config. This is eigenvector-following / ART-style basin exploration,
interleaved with box control across all macro-steps — NOT a separate re-minimize.

Modes are the ones computed at the end of the previous step (same config the next
step starts from) → no extra eigensolve, just retain the vectors.

Design choices (v1):
- `k_vdos_moves: int = 5` new coefficients (1:1 with observed w1..w5). ACT_DIM = 2+k.
- `vdos_move_amp: float` length scale (tune; start small, e.g. 0.01–0.05 reduced
  units). Optionally scale per-mode by 1/√λ_j so softer modes kick more.
- DOF layout CONFIRMED (system.hpp): `sys.x` (2N+2) = `[s_0x,s_0y,...,lnL,gamma]`,
  s_i reduced coords in [0,1)²; `eigvecs_full` returns vectors in the SAME 2N+2 basis
  → `sys.x += amp*c_i*e_i` maps 1:1. Use particle rows e_i[0..2N-1] (renormalized);
  leave box DOF (lnL, gamma) to relax. `amp` is a reduced-coord length (real kick ~
  amp*L); start small (~0.01–0.05 of the reduced mean diameter 1.2/L). Relaxation +
  existing coord wrap handle s_i leaving [0,1).
- PATH-CHAOS implication (from (b)): the quench is basin-chaotic — any finite move
  jumps basins with a broad G distribution. This is WHY VDOS moves are the right
  tool (directed basin selection along the G-governing soft modes, with reward
  feedback), but it makes `vdos_move_amp` the critical knob: too big → explode/scramble
  (fail_pen), too small → no basin change. Sweep amp early. Reward variance stays high,
  so lean on the value-function baseline + large batch to extract the systematic lift.
- Widen the SINGLE Gaussian policy head (reuse PPO + logstd_min/max/ent_coef).
- Quiescence stays on (aP,aS) only, so the agent can "kick then settle".
- Kicks only in held-load macro-steps (not finish-and-measure) in v1.

Edit checklist:
- C++: `env.hpp:29` ACT_DIM; `Env::step` signature → `const VectorXd&`;
  `env.cpp:164` inject displacement; `measure_obs_extras` retain `eigvecs_full`
  vectors in a new `Env::vdos_vecs` member (apply same real-mode filter as w1..w5);
  `rollout.cpp:128` `env.step(a)`; `pybind_module.cpp` step binding → VectorXd.
- Python: `policy.py:14` ACT_DIM = 2 + cfg.k_vdos_moves; `torch_backend.py:48`
  act_dim default; `cem.py:62,66` pass act_dim; `config.py` new params
  (`k_vdos_moves`, `vdos_move_amp`, gate); `learn.py:206-207` keep [:,0:2] metrics.
  storage.py / rollout.py:97 auto-adapt via ACT_DIM.
- Rebuild _core (jamrl env, absolute path). New ACT_DIM ⇒ fresh campaign (no weight
  transfer). Old ACT_DIM=2 and new npz cannot coexist.
- Validate: short run confirms kicks change G more than box-only (widen G-spread
  the agent can reach); watch fail_pen rate (amp too big → blowups).

### Fallback: particle SWAPS (only if VDOS moves don't move G).
Swap radii/identities of particle pairs then re-minimize (swap-MC). Bigger lever on
the contact network, but combinatorial: needs per-particle observations + a
selection head (GNN-ish actor) and a re-minimize+Hessian per swap. Defer until the
VDOS-move experiment reports back. If (c) plateaus at a similar ceiling, this is the
next escalation.
