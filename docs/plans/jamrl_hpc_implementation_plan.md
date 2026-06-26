# `jamrl` — HPC Implementation Plan
### Distributed reinforcement learning of box-control jamming protocols on a SLURM cluster

This is a self-contained implementation specification intended to be handed to **Claude Code** for local development. It describes a production system that scales the validated browser prototype (a box-control jamming RL lab: an agent applies compression/shear forces to a periodic shear box while an L-BFGS minimizer relaxes the packing, rewarded for producing denser jammed states than a zero-action null protocol) to large systems (`N = 1024` and beyond) using a C++ numerical core with OpenMP, a Python orchestration layer, a PyTorch learner, and SLURM job arrays with dependencies.

The reader (Claude Code) should implement this **incrementally, phase by phase, with the validation gate for each phase passing before moving on** (see §9). The physics formulas in §3 and the numerical-hazard fixes in §4 are the correctness contract; they are carried over from a prototype that was validated by a 32-test analytic gate, and the C++ port must reproduce them under finite-difference and limiting-case checks.

---

## Table of contents

0. [Goals and architecture](#0-goals-and-architecture)
1. [Repository layout](#1-repository-layout)
2. [Environment and build system](#2-environment-and-build-system)
3. [Physics core (C++)](#3-physics-core-c)
4. [Numerical hazards and required fixes](#4-numerical-hazards-and-required-fixes)
5. [Python package: config, CLI, data, learner](#5-python-package-config-cli-data-learner)
6. [SLURM orchestration](#6-slurm-orchestration)
7. [Data management and storage](#7-data-management-and-storage)
8. [Testing and validation gates](#8-testing-and-validation-gates)
9. [Implementation phases (build order)](#9-implementation-phases-build-order)
10. [Parameter defaults reference](#10-parameter-defaults-reference)
11. [Run recipes](#11-run-recipes)
12. [Optional / stretch features](#12-optional--stretch-features)

---

## 0. Goals and architecture

### 0.1 What the user wants to run

A single submission that:

- **(a)** starts jamming many independent packings in parallel under the current policy,
- **(b)** aggregates the resulting jammed states and saves, for each: box geometry `(L, γ)`, pressure `(P_target, P_internal)`, density `φ`, contact network, and the Hessian matrix,
- **(c)** updates the learning agent from the collected rollouts,
- **(d)** repeats for a configured number of training rounds,

with strong defaults for everything but full `argparse` control, and the heavy physics accelerated in C++ with multithreading.

### 0.2 The actor–learner–postprocess decomposition

The four steps map onto three SLURM job types per training round `r`:

```
                    ┌─────────────────────────────────────────────┐
  policy(r) ───────►│  ROLLOUT ARRAY  (W parallel workers)         │  step (a)
                    │  each worker runs E episodes under policy(r) │
                    │  → trajectories (obs,act,rew) + jammed states │  step (b)
                    └───────────────┬─────────────────────────────┘
                                    │ afterany
                  ┌─────────────────┴──────────────────┐
                  ▼                                     ▼
        ┌───────────────────┐               ┌────────────────────────┐
        │  LEARN  (1 job)    │  step (c)     │  POSTPROCESS (array)   │
        │  aggregate rollouts│               │  diagonalize Hessians, │
        │  PPO/CEM update    │               │  DOS spectra, moduli,  │
        │  → policy(r+1)     │               │  → analysis parquet    │
        │  submits round r+1 │  step (d)     │  (NON-BLOCKING)        │
        └─────────┬──────────┘               └────────────────────────┘
                  │ (self-resubmit)
                  ▼
              policy(r+1) → ROLLOUT ARRAY(r+1) → ...
```

Rationale:

- **Rollouts dominate cost** and are independent → a job array of `W` tasks, each running `E` episodes. This is where node-hours go.
- **The learner is cheap** (the policy/value nets are tiny MLPs; ~10⁴ transitions/round) → one short job. It performs the PPO/CEM update, writes the next policy, and **submits the next round itself**, so the whole campaign perpetuates from a single initial `jamrl submit`.
- **Hessian/DOS/moduli analysis is expensive but not on the learning critical path** (observations are Hessian-free; see §3.7). It runs as a separate dependency in parallel with the learner and may lag arbitrarily far behind training without stalling it.

### 0.3 Where the policy is evaluated

Actors evaluate the policy **inside C++** (a tiny embedded MLP forward pass) so an entire episode — dozens of macro-steps, each with a full L-BFGS relaxation — is a single C++ call with no per-step Python round-trips. Actors only need the policy *mean* network plus the current action standard deviation; they store `(obs, raw_action, reward, done)`. The learner (PyTorch) recomputes log-probabilities and advantages with autograd. This is the standard actor–learner split and keeps the heavy loop entirely in compiled code.

---

## 1. Repository layout

```
jamrl/
├── pyproject.toml                # scikit-build-core build, deps, entry point
├── CMakeLists.txt                # top-level: pulls Eigen, pybind11, Spectra; OpenMP
├── README.md
├── environment.yml               # conda env (alternative to pip/venv)
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/jamcore/
│   │   ├── types.hpp             # Vec, RNG (splitmix64/xoshiro), config structs
│   │   ├── system.hpp            # System: reduced coords s, lnL, gamma, radii
│   │   ├── cell_list.hpp         # Lees-Edwards linked-cell + Verlet list
│   │   ├── evaluate.hpp          # energy/grad/biased enthalpy
│   │   ├── lbfgs.hpp             # minimizer with the hazard fixes (§4)
│   │   ├── env.hpp               # the MDP: step, reward, termination, finish-and-measure
│   │   ├── hessian.hpp           # sparse full Hessian + backbone DOS Hessian
│   │   ├── moduli.hpp            # bulk/shear via Schur complement
│   │   ├── mlp.hpp               # policy forward (tanh MLP) + Gaussian sampling
│   │   └── rollout.hpp           # run_episode / run_episodes_batch (OpenMP)
│   ├── src/                      # .cpp implementations mirroring the headers
│   └── bindings/
│       └── pybind_module.cpp     # exposes _core: System, relax, hessian, run_episodes_batch, ...
├── src/jamrl/                    # the Python package
│   ├── __init__.py
│   ├── config.py                 # dataclass config + YAML + argparse merge
│   ├── cli.py                    # argparse subcommands: submit/rollout/learn/postprocess/...
│   ├── seeding.py                # deterministic per-episode seed derivation + provenance
│   ├── policy.py                 # PyTorch MLP policy + value nets, (de)serialization to npz
│   ├── ppo.py                    # PPO learner (GAE, clip, normalizers)
│   ├── cem.py                    # CEM learner (population, elites, sigma schedule)
│   ├── rollout.py                # rollout worker: calls _core, writes npz + h5
│   ├── learn.py                  # learner job: aggregate → update → write → resubmit
│   ├── postprocess.py            # diagonalize Hessians, DOS, moduli → parquet
│   ├── storage.py                # npz/h5/parquet readers+writers, schema (§7)
│   ├── slurm.py                  # sbatch template rendering, job-id parsing, dependency wiring
│   ├── analyze.py                # status, plotting from parquet
│   └── templates/
│       ├── rollout.sbatch.j2
│       ├── learn.sbatch.j2
│       └── postprocess.sbatch.j2
├── tests/
│   ├── test_evaluate.py          # γ=0 reduction, FD gradient, γ-wrap invariance
│   ├── test_lbfgs.py             # convergence, stall handling, determinism
│   ├── test_env.py               # null jams, action semantics, reward, finish-and-measure
│   ├── test_cells.py             # cell-list == O(N²) brute force; scaling benchmark
│   ├── test_hessian.py           # Hessian·v == FD; PSD; shear-stabilization (G≥0)
│   ├── test_moduli.py            # B,G via Schur; FD cross-check on small N
│   ├── test_rollout.py           # batch runner parity vs Python step loop; determinism
│   └── test_learner.py           # PPO/CEM reward improves on a toy; checkpoint round-trip
└── docs/
    └── physics.md                # the formulas of §3, for reference
```

---

## 2. Environment and build system

### 2.1 Dependencies

C++ (all header-only, fetched via CMake `FetchContent` so there is nothing to install by hand):

- **Eigen 3.4+** — dense/sparse linear algebra, decompositions.
- **pybind11 2.11+** — Python bindings.
- **Spectra 1.0+** — sparse eigensolver (low-frequency DOS modes at large `N`); built on Eigen.
- **OpenMP** — threading (compiler-provided).

Python (pin in `environment.yml` / `pyproject.toml`):

```
numpy>=1.26, scipy>=1.11, h5py>=3.10, pandas>=2.1, pyarrow>=14,
pyyaml>=6, torch>=2.1 (CPU build is fine; nets are tiny),
pybind11>=2.11, scikit-build-core>=0.8, cmake>=3.27, ninja>=1.11,
pytest>=8, matplotlib>=3.8
```

### 2.2 `pyproject.toml` (scikit-build-core)

```toml
[build-system]
requires = ["scikit-build-core>=0.8", "pybind11>=2.11"]
build-backend = "scikit_build_core.build"

[project]
name = "jamrl"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy","scipy","h5py","pandas","pyarrow","pyyaml","torch","matplotlib"]

[project.scripts]
jamrl = "jamrl.cli:main"

[tool.scikit-build]
cmake.version = ">=3.27"
build-dir = "build/{wheel_tag}"
wheel.packages = ["src/jamrl"]
```

### 2.3 CMake essentials

The top-level `CMakeLists.txt` should: enable C++17, find OpenMP (`find_package(OpenMP REQUIRED)`), `FetchContent` Eigen + pybind11 + Spectra, build the extension `jamrl._core` from `cpp/`, and link `OpenMP::OpenMP_CXX`. Compile with `-O3 -march=native -funroll-loops` (gate `-march=native` behind an option `JAMRL_NATIVE=ON` default, since cluster login nodes may differ from compute nodes — see §6.7). Expose `-DJAMRL_USE_SPECTRA=ON`.

Local install for development:

```bash
pip install -e . --no-build-isolation   # rebuilds _core on edit
# or: cmake -S . -B build && cmake --build build -j   (for C++-only iteration)
```

---

## 3. Physics core (C++)

All formulas below are the **correctness contract**. The system is 2D bidisperse harmonic disks (50:50 mixture, radii `R = 0.5` and `0.7`), energy and mass units `ε = m = 1`. Implement the `O(N²)` version first for correctness (Phase 2), then accelerate with cell lists (Phase 4) and require the accelerated path to match the brute-force path to `1e-12`.

### 3.1 State and box

State vector `x` of dimension `2N + 2`:

```
x = [ s_0x, s_0y, s_1x, s_1y, ..., s_(N-1)x, s_(N-1)y,  ℓ,  γ ]
```

- `s_i ∈ [0,1)²` are **reduced** (fractional) coordinates.
- `ℓ = ln L`, where `L` is the box side; area `A = L²`.
- `γ` is the Lees-Edwards shear strain.
- Deformation matrix `h = L · [[1, γ], [0, 1]]`, so a real-space position is `h · s_i` and a real separation is `h · Δs`.

Frozen radii `R_i` and target pressure `P` live on the `System`.

### 3.2 Sheared minimum image and `evaluate`

For each pair `(i, j)` compute the sheared minimum-image separation:

```
dsy = s_iy - s_jy ;  dsy -= round(dsy)
u   = (s_ix - s_jx) + γ*dsy ;  u -= round(u)
rx  = L*u ;  ry = L*dsy ;  r = sqrt(rx*rx + ry*ry)
σij = R_i + R_j
```

A contact exists iff `r < σij`. For harmonic disks, with overlap `δ = 1 - r/σij`:

```
E_pair = 0.5 * δ²                       # pair energy
V'     = -δ / σij                       # dV/dr   (note V'<0 in contact)
V''    = 1 / σij²                       # constant for harmonic
n      = (rx, ry) / r                   # unit contact normal
```

Accumulate (let `sumVr = Σ V'·r`, `Eγ = Σ V'·n_x·r_y`, `E = Σ E_pair`):

```
# force on reduced coords (gradient of E wrt s):
gx_i +=  V' * n_x * L ;   gx_j -= V' * n_x * L
gy_i +=  V' * (γ*n_x + n_y) * L ;  gy_j -= ...

P_int          = -sumVr / (2*A)
dH/dℓ  (unbias)=  sumVr + 2*P*A
dH/dℓ  (biased)=  sumVr + 2*(P+ΔP)*A          # NO shear coupling — see §4.3
dH/dγ  (unbias)=  Eγ
dH/dγ  (biased)=  Eγ - σF                     # σF frozen per macro-step — see §4.3

H   = E + P*A
H_b = E + (P+ΔP)*A - σF*γ
```

The **unbiased** infinity-norm `|∇H|_∞` over all `2N+2` components is the convergence criterion (see §3.4). `ΔP` and `σF` are the applied loads (zero in the unbiased evaluate). `maxOv = max δ` is tracked for failure detection.

`evaluate` should fill a result struct `{ E, H, H_b, grad[2N+2], fInf_unbiased, fInf_biased, P_int, maxOv, Eγ }`.

### 3.3 `γ`-wrap (exact periodicity)

`E` is exactly periodic in `γ` with period 1 (the sheared lattice is the same lattice under a coordinate relabel). Wrap after each macro-step:

```
k = round(γ)
if k != 0:
    for all i: s_ix += k * s_iy      # exact energy-invariant relabeling
    γ -= k
for all i: s_ix -= floor(s_ix); s_iy -= floor(s_iy)
```

The test in §8 must confirm energy is invariant under this relabeling to machine zero.

### 3.4 L-BFGS relaxer

Two-loop recursion, history `m = 8`, operating on the full `x` (positions + box dofs jointly) under the **biased** enthalpy `H_b(ΔP, σF)`. Backtracking Armijo line search. This minimizer carries three required fixes from §4 (numeric Armijo slack, memory-reset + step-scale-reset on line-search failure, and a stall counter). API:

```cpp
struct LBFGSParams { int memory=8; double c1=1e-4; int max_ls=30; int stall_max=6; };
struct RelaxResult { int iters; int n_eval; bool stalled; EvalResult ev; };

// relax in place under biased enthalpy; runs up to n_steps L-BFGS iterations.
RelaxResult lbfgs_relax(System& sys, double dP, double sigF,
                        int n_steps, const LBFGSParams& p);
```

Convergence (unbiased) is `fInf_unbiased < ftol_eff` AND `|P_int - P|/P < ptol`, where
`ftol_eff = max(ftol_abs, ftol_rel_P · P · σ̄)` with `σ̄ = 1.2` (mean diameter), `ftol_abs = 1e-10`, `ftol_rel_P = 1e-5`, `ptol = 1e-4`. While a load is held, the relaxer settles toward the *biased* state (`P_int → P+ΔP ≠ P`), so the unbiased criterion cannot be met until the agent releases — termination is itself a learned decision.

### 3.5 The MDP environment

One macro-step: the agent picks `(a_P, a_σ) ∈ [-1,1]²`; this defines the loads held during `n_relax` L-BFGS iterations; then unbiased quantities are read for the observation, reward, and termination test.

```
Peff = max(0.1*P, P*(1 + κ_P * a_P))         # PRESSURE FLOOR — see §4.2
ΔP   = Peff - P
A0   = L²  (frozen at the start of this macro-step)
σF   = κ_σ * P * A0 * a_σ                     # generalized force on γ — see §4.3

lbfgs_relax(sys, ΔP, σF, n_relax, params)     # fresh history each macro-step
γ-wrap(sys)
evaluate(sys, ΔP=0, σF=0)                      # unbiased view
t += 1
```

Observation vector (10 features, all intensive → **N-independent**, so a policy can in principle transfer across system sizes):

```
o[0] = clamp((log10(fInf_unbiased + 1e-16) + 14)/15, 0, 1)
o[1] = tanh(((P_int - P)/P)/5)
o[2] = tanh((Eγ/(P*A))/5)            # shear-stress residual
o[3] = tanh(2*γ)
o[4] = (φ - 0.8)*10
o[5] = z>0 ? tanh(5*(z/z_iso - 1)) : -1
o[6] = tanh(20*maxOv)
o[7] = t / T_cap
o[8] = prev_a_P
o[9] = prev_a_σ
```

Termination and reward (the order of these checks matters):

```
quiet = (max(|a_P|,|a_σ|) < quiesce_tol) ? quiet+1 : 0
φ_now = density(sys)

if  !isfinite(H) or maxOv > 0.9 or φ_now < 0.3:        # MELT/OVERLAP/BLOWUP — §4.2
        reward -= fail_pen ; done = true
        outcome ∈ {blowup, overlap, melt}
elif converged (unbiased):                              # agent released and it jammed
        reward += w_φ * (φ - φ_null[seed]) ; done = true ; outcome = converged
elif t >= T_cap or quiet >= quiesce_n:                  # FINISH-AND-MEASURE
        cap = min(60000, max(finish_cap, round(3e2/sqrt(P))))   # low-P slowing — §4.4
        run unbiased L-BFGS up to `cap` iters (break on stall or convergence)
        if converged:
            reward += w_φ*(φ - φ_null[seed]) - trunc_pen
            outcome = (quiet>=quiesce_n) ? quiesced : capped
        else:
            reward -= fail_pen ; outcome = unfinished
        done = true
reward -= c_step      # per macro-step cost, every step
```

`φ_null[seed]` is the density reached by a **zero-action episode** on the same seed (computed once per `(N,P,seed)` and cached; see §5.5). The outcome taxonomy `{converged, capped, quiesced, overlap, melt, blowup, unfinished}` is recorded per episode for failure-mode monitoring.

Expose both a fully-internal `step` (for tests/debug, mirrors the prototype) and the batch runner of §3.8 (for production rollouts).

### 3.6 Hessian (sparse, full `2N+2`)

Assemble the **unbiased** enthalpy Hessian over `(s, ℓ, γ)` as Eigen triplets → `SparseMatrix<double>`. Block structure (with `h` the deformation matrix, `K = V''·nnᵀ + (V'/r)(I − nnᵀ)` the real-space contact stiffness):

- `ss` block: `hᵀ K h`, assembled with the usual `+` for `i`, `−` for `j` pattern.
- `s–ℓ` coupling: `w = (V''·r + V')·(nᵀh)`.
- `s–γ` coupling: `d/dγ[ V'·(nᵀh)_a ]`, expand using `dr/dγ = n_x·r_y`, the normal derivative `dn = ((r_y,0) − n·(dr/dγ))/r`, and the explicit `h_γ` term (`(nᵀh_γ)_y = n_x·L`).
- `ℓℓ = 4·P·A + Σ(V''·r² + V'·r)`.
- `ℓγ = Σ(V''·r + V')·n_x·r_y`.
- `γγ = E_γγ`.

The block formulas are intricate; the **test gate (§8) is the source of truth**: `Hessian·v` must match a central finite difference of the analytic gradient to relative `1e-9`, at both a random `γ≠0` configuration and a relaxed jammed state, and the matrix must be symmetric to `1e-12`. Do not proceed past Phase 5 until this passes.

For DOS, also assemble the **backbone position Hessian**: fixed box (positions only), restricted to the rattler-removed set (particles with ≥3 contacts among the kept set, iterated to a fixed point), dimension `2·n_keep`. This is the `K`-only block with no box dofs.

### 3.7 Moduli (Schur complement)

From the full unbiased Hessian `H_full`:

```
B (bulk)  = (1/(4A)) * [ H_ℓℓ - H_ℓq · H_qq⁺ · H_qℓ ]    # eliminate q = (s, γ)
G (shear) = (1/A)    * [ H_γγ - H_γq · H_qq⁺ · H_qγ ]    # eliminate q = (s, ℓ)
```

`H_qq⁺` is applied by solving `H_qq y = b` with a sparse Cholesky (`Eigen::SimplicialLDLT`) on the kept block; if it is singular at jamming (it can be, due to floating zero modes), fall back to a conjugate-gradient pseudo-inverse with a small Tikhonov regularizer `1e-9`. Because `γ` is now a *minimization* dof, every converged packing satisfies `dH/dγ = 0` with non-negative curvature → these are **shear-stabilized** packings and `G ≥ 0` by construction (unlike the soft-mode prototype where `G<0` on a majority of packings). The test gate must confirm `G ≥ -1e-8` on null-protocol jammed states.

### 3.8 Policy MLP and the batch episode runner

Embed a fixed-architecture tanh MLP forward pass:

```cpp
struct Policy {
    // obs_mean,obs_std (length obs_dim) for normalization (must match learner!)
    // W0,b0 -> hidden0; W1,b1 -> hidden1; Wmu,bmu -> action mean (act_dim)
    // log_std (length act_dim)
};
// returns action mean μ given (normalized) obs
```

The action is sampled `a_raw = μ + exp(log_std) ⊙ ξ`, `ξ ~ N(0,1)` from the **episode RNG**; the env uses `clip(a_raw, -1, 1)` for dynamics, but the **stored** action is `a_raw` (so the learner can recompute the Gaussian log-prob for the PPO ratio). The batch runner:

```cpp
struct EpisodeOut {
    // trajectory:
    MatrixXd obs;     // [T, obs_dim]
    MatrixXd act;     // [T, act_dim]   (raw, pre-clip)
    VectorXd rew;     // [T]
    // final jammed state (only if outcome ∈ {converged,capped,quiesced}):
    bool jammed; int outcome_code;
    VectorXd x_final;          // 2N+2
    double L, gamma, P_int, phi; int z_int, n_rattlers; double dz;
    // contact list (i,j) for kept pairs; sparse Hessian triplets if requested
    // moduli if requested (B,G); eigen-spectrum if requested
};

std::vector<EpisodeOut>
run_episodes_batch(const System& proto, const Policy& pol,
                   const std::vector<uint64_t>& seeds,
                   const EnvConfig& cfg, const SaveFlags& save,
                   int parallel_mode /*0=episode,1=intra*/);
```

`parallel_mode = episode` (default): `#pragma omp parallel for schedule(dynamic)` over `seeds`, each episode **single-threaded** (Eigen/BLAS set to 1 thread). This gives near-linear node scaling **and bitwise determinism per seed**. `parallel_mode = intra`: one episode at a time with OpenMP inside `evaluate`/Hessian (for very large `N` where a single evaluate is itself expensive); note this **breaks bitwise determinism** (reduction order) and should be documented as such.

### 3.9 pybind11 surface (`jamrl._core`)

Expose: `make_system(N, seed, phi0, P) -> System`; `evaluate(sys, dP, sigF) -> dict`; `relax(sys, dP, sigF, n_steps, params) -> dict`; `step(sys, aP, aS, cfg) -> dict` (debug); `hessian_sparse(sys) -> (data, indices, indptr, shape)`; `hessian_dos(sys) -> (...)`; `eigvals_dos(sys, k=None) -> ndarray` (dense if `k is None`, else Spectra lowest-k); `bulk_modulus(sys)`, `shear_modulus(sys)`; `run_episodes_batch(...)` returning a list of dict/`EpisodeOut`. Pass numpy arrays zero-copy via `Eigen::Ref` / the buffer protocol. Provide `set_num_threads(n)` and `eigen_set_threads(n)`.

---

## 4. Numerical hazards and required fixes

These four issues were discovered and fixed in the prototype; **they will recur in the C++ port** and must be implemented from the start. Each has a test in §8.

### 4.1 L-BFGS Armijo sub-precision spin

Near convergence the Armijo sufficient-decrease margin `c1·a·(∇H_b·d)` can fall below the floating-point resolution of `H_b`, so no step length ever "decreases" the objective and the backtracker spins forever (in the prototype this manifested as ~5×10⁶ evaluations per episode, ~90 s). **Fix:** accept a step when `H_b(x_new) ≤ H_b(x) + c1·a·gd + slack`, with `slack = 1e-14·max(1e-30, |H_b|)`. On a *failed* line search, reset the L-BFGS history and the initial step scale `a0 = 1`, and increment a `stall` counter; after `stall_max = 6` consecutive failures, mark the relaxer stalled and stop (the inner relax loop and the finish-and-measure loop both break on stall). Expose `stalled`.

### 4.2 Box runaway → pressure floor + melt detection

With full decompression (`a_P = -1`, `κ_P = 1`) the naive effective pressure is zero, removing all confinement; the box then expands without bound (`L ~ 10¹⁴³` was observed). **Fix:** floor the effective pressure, `Peff = max(0.1·P, P·(1 + κ_P·a_P))`, so full decompression means "anneal at one-tenth pressure," never zero confinement. Independently, detect runaway by density: if `φ < 0.3` at any macro-step, end the episode with outcome `melt` and the failure penalty.

### 4.3 Joint `γ`/`ℓ` runaway → shear as a frozen generalized force

If shear is applied as a stress coupled to the *current* area (`-σ·A·γ` with live `A`), then when `σγ > P_eff` the `γ` and `ℓ` dofs co-amplify and diverge. **Fix:** treat the shear input as a **generalized force conjugate to `γ`** (units of energy), frozen at the start of each macro-step: `σF = κ_σ·P·A₀·a_σ` with `A₀ = L²` captured before relaxation. The biased enthalpy is `H_b = E + (P+ΔP)L² − σF·γ`, so `∂H_b/∂ℓ` has **no shear term** and `∂H_b/∂γ = E_γ − σF`. This decouples the box-size and shear drives and keeps the box confined.

### 4.4 Low-pressure critical slowing → pressure-scaled finish budget

At low target pressure the relaxation slows critically; a fixed iteration cap leaves states "not jammed." **Fix:** scale the finish-and-measure budget as `cap = min(60000, max(finish_cap, round(3e2/√P)))`. (In the prototype this restored agreement of low-`P` jammed densities with the reference to five digits.)

### 4.5 Singular Hessian at jamming (moduli/DOS)

The unbiased Hessian has two trivial zero modes (global translations) and may be numerically singular. For **DOS**, work in the rattler-removed backbone and drop the two lowest (≈0) eigenvalues before binning `ω = √max(0, λ)`. For **moduli**, use `SimplicialLDLT` on the eliminated block with a CG/Tikhonov fallback (§3.7). Never invert the full Hessian directly.

### 4.6 Determinism policy

Bitwise reproducibility is guaranteed only in `parallel_mode = episode` with one thread per episode (fixed reduction order). Each episode's RNG is seeded deterministically from `(campaign_seed, round, worker, episode_index)` (§5.2). Document that `parallel_mode = intra` and any multi-threaded BLAS reductions sacrifice bitwise determinism for speed.

---

## 5. Python package: config, CLI, data, learner

### 5.1 Config system (defaults + YAML + CLI)

A single frozen `dataclass` `Config` holds every parameter with a strong default (§10). Resolution order, lowest to highest precedence:

1. dataclass defaults (in code),
2. an optional YAML file (`--config run.yaml`),
3. explicit CLI flags (every field is also an `argparse` argument; flags win).

```python
# jamrl/config.py
@dataclass(frozen=True)
class Config:
    # system
    N: int = 1024; P: float = 1e-3; phi0: float = 0.80
    # agent loading
    kappa_P: float = 1.0; kappa_sigma: float = 0.5
    n_relax: int = 20; T_cap: int = 60
    # reward
    w_phi: float = 400.0; c_step: float = 0.01
    fail_pen: float = 2.0; trunc_pen: float = 0.5
    quiesce_tol: float = 0.05; quiesce_n: int = 3
    finish_cap: int = 12000
    # tolerances
    ftol_abs: float = 1e-10; ftol_rel_P: float = 1e-5; ptol: float = 1e-4
    # learner
    algo: str = "ppo"               # ppo | cem
    lr: float = 3e-4; gamma: float = 0.995; lam: float = 0.95
    clip: float = 0.2; ppo_epochs: int = 6; minibatch: int = 1024
    ent_coef: float = 3e-3; vf_coef: float = 0.5; logstd_init: float = -0.5
    hidden: tuple = (64, 64)
    cem_pop: int = 64; cem_elite_frac: float = 0.25; cem_sigma0: float = 0.3
    cem_eps_per_cand: int = 4
    # campaign / parallelism
    rounds: int = 1000
    workers: int = 64               # SLURM array size per round
    episodes_per_worker: int = 8
    eval_seeds: tuple = tuple(range(101, 107))
    parallel_mode: str = "episode"  # episode | intra
    threads_per_task: int = 16
    # data
    save_hessian: str = "sparse"    # none | spectrum | sparse | dense
    hessian_stride: int = 1         # save every k-th jammed state's Hessian
    compression: str = "gzip"
    # slurm
    partition: str = ""; account: str = ""; time_rollout: str = "02:00:00"
    time_learn: str = "00:20:00"; time_post: str = "01:00:00"
    mem_rollout: str = "8G"; mem_learn: str = "8G"; mem_post: str = "16G"
    dependency_mode: str = "afterany"   # afterany (robust) | afterok (strict)
    min_worker_frac: float = 0.6        # learner aborts if fewer rollouts present
    # misc
    seed: int = 12345
    campaign_root: str = "./campaigns"
    name: str = "run"
```

`config.py` provides `add_arguments(parser)` (auto-generates flags from the dataclass), `from_args(args)`, `to_yaml()/from_yaml()`, and a `config_hash()` (sha1 of the canonical YAML) for provenance.

### 5.2 Deterministic seeding and provenance

```python
# jamrl/seeding.py
def episode_seed(campaign_seed, rnd, worker, ep_idx) -> int:
    # splitmix64-style mixing → 64-bit seed; reproducible & well-spread
def write_provenance(campaign_dir, config):
    # git commit (subprocess), config_hash, hostname, python/torch/numpy versions,
    # OMP_NUM_THREADS, timestamp → provenance.json
```

Seeds are stored alongside trajectories so any state can be recomputed bit-for-bit.

### 5.3 CLI subcommands (`jamrl/cli.py`)

```
jamrl submit       [--config ...] [flags] [--resume] [--test-only]
                   # resolve config, create/append campaign dir, write initial policy,
                   # submit round 0 (rollout array + learn + postprocess). Self-perpetuates.
jamrl run-local    [...]                 # whole loop in ONE process (multiprocessing rollouts); no SLURM
jamrl rollout      --campaign DIR --round R --worker K   # one array task
jamrl learn        --campaign DIR --round R              # aggregate → update → write → resubmit
jamrl postprocess  --campaign DIR --round R [--shard S --nshards M]
jamrl status       --campaign DIR        # rounds done, reward curve tail, last eval
jamrl plot         --campaign DIR        # regenerate figures from parquet
```

### 5.4 Policy/value nets and (de)serialization (`jamrl/policy.py`)

PyTorch modules: `PolicyNet(obs_dim, hidden, act_dim)` → action mean, plus a learned `log_std` parameter vector; `ValueNet(obs_dim, hidden)` → scalar. A `RunningNorm` (Welford mean/var) for observations. Serialize the **actor-facing** subset to npz for the C++ runner:

```python
def save_policy_npz(path, policy, obs_norm):
    # W0,b0,W1,b1,Wmu,bmu,log_std, obs_mean,obs_std   (float64)
def load_policy_npz(path) -> dict   # consumed by rollout.py → _core.Policy
```

The learner checkpoint (`.pt`) holds policy + value + optimizer + `RunningNorm` state + RNG, for exact resume.

### 5.5 Rollout worker (`jamrl/rollout.py`)

For round `r`, worker `k`:

1. Load `policy/round_{r:04d}.npz`; build a `_core.Policy`.
2. Ensure `φ_null[seed]` is available for every seed this worker will use — null densities are cached per `(N,P,seed)` in `campaign/null_cache.h5`; compute any missing via zero-action episodes (also in `_core`, which the env uses internally for the reward; the worker just needs the cache populated, so a small helper computes and stores them under a file lock).
3. Derive `episodes_per_worker` seeds via `episode_seed(...)`.
4. Call `_core.run_episodes_batch(proto, policy, seeds, env_cfg, save_flags, parallel_mode)` with `threads_per_task` threads.
5. Write trajectories to `rollouts/round_{r}/worker_{k}.npz` and jammed states to `states/round_{r}/worker_{k}.h5` (schema §7). Honor `save_hessian` / `hessian_stride`.

A blown-up episode is recorded as a failure outcome; it never crashes the worker.

### 5.6 PPO learner (`jamrl/ppo.py`)

Aggregate all `rollouts/round_{r}/worker_*.npz` (tolerate missing files; require ≥ `min_worker_frac` of `workers`, else exit non-zero with a clear message so the chain halts visibly). Then:

- Update the observation normalizer from the round's observations (Welford), persist it.
- Compute returns/advantages with **GAE(λ)**; every episode boundary is a true terminal (no bootstrapping across episodes).
- PPO clipped-surrogate objective with the value loss and an entropy bonus; `ppo_epochs` passes over minibatches; clip `0.2`; Adam at `lr`. The learned `log_std` is a parameter (initialized `logstd_init`).
- Write `policy/round_{r+1:04d}.npz` (actor subset) and `checkpoints/learner_round_{r+1:04d}.pt`.
- Run a quick **greedy** eval on `eval_seeds` (deterministic actions = policy mean), compute `⟨φ − φ_null⟩`, success rate, mean `|a_P|`/`|a_σ|`, and (for `N ≤ 64`, or whenever cheap) `B,G`, `Δz`, rattler/shear-stable fractions; append a row per metric to `analysis/summary.parquet`.

Hyperparameter defaults mirror the validated prototype: `gamma 0.995, lam 0.95, clip 0.2, lr 3e-4, ent_coef 3e-3, logstd_init -0.5`.

### 5.7 CEM learner (`jamrl/cem.py`)

Alternative to PPO, selected by `--algo cem`. The "policy" for a round is a population of `cem_pop` flattened parameter vectors sampled from the current search distribution `N(μ, diag(σ²))`. Map candidates to workers (each worker evaluates a contiguous slice of candidates, `cem_eps_per_cand` episodes each, **every candidate scored on the same seed set** for low-variance comparison). The learner ranks candidates by mean return, refits `μ` to the elite mean and `σ` to the elite std (with a small floor), and ships the new distribution's mean as `policy/round_{r+1}.npz` (so eval/visualization always has a concrete policy). CEM needs no value net and no gradients.

### 5.8 Learner self-resubmission (step d)

At the end of the learn job, if `r+1 < rounds` and no `STOP` sentinel file exists in the campaign dir, the learner calls `jamrl.slurm.submit_round(campaign, r+1)` (via `sbatch`, see §6) to launch the next rollout array + learn + postprocess. Otherwise it writes a `DONE` sentinel. This is what makes the campaign perpetuate from one initial submission, and `touch STOP` is a clean way to end it after the current round.

---

## 6. SLURM orchestration

### 6.1 The per-round dependency graph

```
submit_round(r):
    roll_jid = sbatch --array=0-(W-1) rollout.sbatch  r          # step (a)+(b)
    learn_jid = sbatch --dependency=<DEP>:roll_jid  learn.sbatch r   # step (c)+(d)
    post_jid  = sbatch --dependency=<DEP>:roll_jid  postprocess.sbatch r  # non-blocking analysis
    record {roll_jid, learn_jid, post_jid} → rounds/round_{r}.json
```

`<DEP>` is `afterany` by default (robust to a few failed array tasks; the learner checks it has ≥ `min_worker_frac` data) or `afterok` for strict mode. The learner's own resubmission (§5.8) extends the chain, so a single `jamrl submit` launches the entire campaign.

### 6.2 `slurm.py` responsibilities

- Render `templates/*.sbatch.j2` with the resolved config (partition, account, time, mem, cpus-per-task = `threads_per_task`, array size = `workers`, env vars).
- `sbatch` via `subprocess`, parse the returned job id (`Submitted batch job 12345` → `12345`). For arrays, the base job id works with `afterany:JID`/`afterok:JID` to wait on the whole array.
- `submit_round(campaign, r)` wiring the three jobs with dependencies.
- `--test-only` passthrough (`sbatch --test-only`) for a dry run that validates scripts and dependencies without queueing work.

### 6.3 `templates/rollout.sbatch.j2`

```bash
#!/bin/bash
#SBATCH --job-name=jamrl-roll-r{{ round }}
#SBATCH --array=0-{{ workers_minus_1 }}
#SBATCH --cpus-per-task={{ threads_per_task }}
#SBATCH --mem={{ mem_rollout }}
#SBATCH --time={{ time_rollout }}
{% if partition %}#SBATCH --partition={{ partition }}{% endif %}
{% if account %}#SBATCH --account={{ account }}{% endif %}
#SBATCH --output={{ campaign }}/logs/roll_r{{ round }}_%A_%a.out

export OMP_NUM_THREADS={{ threads_per_task }}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1     # episode-parallel: no nested BLAS threads
export MKL_NUM_THREADS=1
srun jamrl rollout --campaign {{ campaign }} --round {{ round }} \
                   --worker ${SLURM_ARRAY_TASK_ID}
```

(In `parallel_mode = intra`, set `OPENBLAS_NUM_THREADS={{ threads_per_task }}` and run episodes serially — `rollout.py` already branches on the mode.)

### 6.4 `templates/learn.sbatch.j2`

```bash
#!/bin/bash
#SBATCH --job-name=jamrl-learn-r{{ round }}
#SBATCH --cpus-per-task=4
#SBATCH --mem={{ mem_learn }}
#SBATCH --time={{ time_learn }}
{% if partition %}#SBATCH --partition={{ partition }}{% endif %}
{% if account %}#SBATCH --account={{ account }}{% endif %}
#SBATCH --output={{ campaign }}/logs/learn_r{{ round }}_%j.out

export OMP_NUM_THREADS=4
srun jamrl learn --campaign {{ campaign }} --round {{ round }}
# (jamrl learn submits round r+1 at its end unless STOP exists)
```

### 6.5 `templates/postprocess.sbatch.j2`

Optionally an array over shards of the round's states (dense diagonalization at `N=1024` is ~seconds/state; shard for throughput). It reads `states/round_{r}/*.h5`, computes DOS spectra (dense LAPACK, or Spectra lowest-k for large `N`), participation ratios, and moduli if not already stored, and writes `analysis/round_{r}/spectra_shard_{s}.npz`, then appends aggregated scalars to `analysis/summary.parquet`. Because it only reads states, it never blocks `learn`.

### 6.6 Robustness

- **Partial failures:** with `afterany`, the learner aggregates whatever rollouts landed; below `min_worker_frac` it exits non-zero (chain visibly halts; `jamrl status` shows where). Optionally it can submit a small "top-up" rollout array for the missing workers before proceeding.
- **Resume:** `jamrl submit --resume` detects the highest fully-written `policy/round_XXXX.npz` + matching checkpoint and continues from there; round submission is idempotent (skip a round whose outputs already exist).
- **Stop cleanly:** `touch <campaign>/STOP`; the next learner writes `DONE` instead of resubmitting.
- **Requeue:** array tasks are stateless and safe to requeue; each writes its own worker file atomically (write to `*.tmp`, then `os.replace`).

### 6.7 Resource and scaling guidance

- Build the extension on a **compute node** (or with `JAMRL_NATIVE=OFF`) so `-march=native` matches the run architecture; mismatched ISA between login and compute nodes causes `SIGILL`.
- Rollout tasks: `cpus-per-task = threads_per_task` (default 16); memory is modest (trajectories + a few sparse Hessians); time = `episodes_per_worker × per-episode wall`. Benchmark per-episode wall at `N=1024` in Phase 4 to size `time_rollout`.
- Throughput knobs: total episodes/round = `workers × episodes_per_worker`. Scale `workers` (array size) for more parallelism; keep `episodes_per_worker` ≥ 4 so OpenMP-over-episodes stays well-fed.
- Learner: tiny; 4 CPUs is plenty. GPU is unnecessary for these MLPs (offer `--device cuda` as an option but default CPU).
- Post-processing memory scales with dense eig at large `N` (`mem_post`); switch to Spectra lowest-k (`save_hessian=spectrum`, `--dos-k 60`) above `N ≈ 2048` to avoid `O(N³)` dense costs.

---

## 7. Data management and storage

### 7.1 Layout

```
campaigns/<name>/
├── config.yaml                 # resolved config
├── provenance.json             # git hash, config hash, env, versions
├── null_cache.h5               # φ_null per (N,P,seed), file-locked
├── policy/round_0000.npz ...   # actor-facing weights per round
├── checkpoints/learner_round_0000.pt ...
├── rollouts/round_0000/worker_000.npz ...   # RL transitions (small)
├── states/round_0000/worker_000.h5 ...      # jammed states + Hessian (large)
├── analysis/
│   ├── round_0000/spectra_shard_000.npz ...
│   └── summary.parquet         # one row per (round, metric) for plotting
├── rounds/round_0000.json      # the round's SLURM job ids + status
├── logs/...                    # SLURM stdout/stderr
└── STOP / DONE                 # sentinels
```

### 7.2 Schemas

**Trajectory npz** (`rollouts/.../worker_k.npz`), small, fast to load:
`obs [ΣT, 10] f32`, `act [ΣT, 2] f32` (raw pre-clip), `rew [ΣT] f32`, `done [ΣT] bool`, `ep_ptr [E+1] i32` (episode boundaries), `seeds [E] u64`, `outcome [E] i8`, `phi [E] f32`, `phi_null [E] f32`, `steps [E] i16`.

**Jammed-state HDF5** (`states/.../worker_k.h5`), one group `/ep{j}` per *successfully jammed* episode:
- attrs: `seed, outcome, L, gamma, P_target, P_int, phi, z, z_iso, dz, n_keep, n_rattlers, n_contacts`.
- datasets: `s [N,2] f32` (reduced coords), `radii [N] f32`, `contacts [n_contacts,2] i32`.
- if `save_hessian == sparse`: `H_data, H_indices, H_indptr` (CSR, f64/i32) + attr `H_shape` — store only on `j % hessian_stride == 0`.
- if `save_hessian == spectrum`: `eig [2*n_keep] f32` (backbone DOS spectrum) — the cheap default for long campaigns.
- if `save_hessian == dense`: `H [2N+2, 2N+2] f32` (only sensible for diagnostics at small `N`).
All datasets gzip-compressed.

**Summary parquet** (`analysis/summary.parquet`), columns: `round, episodes, mean_reward, eval_dphi, eval_success, eval_cost_kevals, mean_absaP, mean_absaS, mean_absgamma, Bbar, Gbar, dzbar, rattler_frac, shear_stable_frac, omega_star, sigma_policy, wall_seconds, git_hash`. Append-only; `jamrl plot`/`status` read this.

### 7.3 Volume control

Sparse Hessian at `N=1024` is ≈0.2–0.5 MB/state; at `workers·episodes_per_worker ≈ 512` jammed states/round that is ≈150–250 MB/round, ≈150–250 GB over 1000 rounds. **Default to `save_hessian=spectrum`** (≈16 KB/state → ≈8 GB/1000 rounds) for long campaigns and reserve `save_hessian=sparse` (full matrices) for shorter, analysis-focused runs or via `hessian_stride > 1`. Document this prominently; the user asked for Hessian data, but the storage trade-off must be a conscious, configurable choice. Provide `jamrl compact --keep-spectrum --round-range a:b` to down-convert old rounds' full Hessians to spectra and reclaim space.

---

## 8. Testing and validation gates

Port the prototype's analytic gate to `pytest`. These are **gates** — a phase is not done until its tests pass. All comparisons use the C++ core via `_core`.

`test_evaluate.py`
- `γ=0` reduces exactly to the isotropic engine: with `γ=0`, `H` and the `(s,ℓ)` gradient match a separate fixed-box reference (or a reduced-form recomputation) to `1e-13`; `∂H/∂γ` equals the `E_γ` formula.
- FD gradient over all `2N+2` dofs at `γ=0.2`, both unbiased and biased (`ΔP, σF ≠ 0`), to relative `< 1e-5`.
- `γ`-wrap relabeling: energy invariant to machine zero; `γ` lands in `(-½, ½]`.

`test_lbfgs.py`
- Random init at small `N` converges to `fInf_unbiased < ftol_eff` and `|P_int−P|/P < ptol`.
- The Armijo-slack fix: an episode near convergence uses `O(10³)` evaluations, not `O(10⁶)` (assert an evaluation-count ceiling).
- Determinism: identical seeds → identical trajectories (bitwise) in `parallel_mode=episode`.

`test_env.py`
- Zero-action ("null") episodes jam at seeds `1,2,101` with the expected densities (`≈0.845/0.824/0.841` at `N=32, P=1e-3` — these are the prototype reference values; use them as regression anchors at `N=32`).
- Action semantics: held `a_P=+1` drives `(P_int−P)/P → ≈ κ_P`; held `a_σ=+1` drives `γ>0` early, then yields past the wrap under sustained stress.
- Pressure floor + melt: `a_P=-1` does **not** blow up `L`; an over-decompression sequence ends as `melt`, not `unfinished`/`blowup`.
- Finish-and-measure: releasing after a load reaches an unbiased jammed state (`outcome=converged`).

`test_cells.py`
- Cell/Verlet-list `evaluate`, gradient, and Hessian match the brute-force `O(N²)` path to `1e-12` at `N ∈ {64, 256, 1024}` and several `γ`.
- Scaling benchmark: per-`evaluate` time vs `N` shows sub-quadratic growth (record numbers in `docs/physics.md`).

`test_hessian.py`
- `Hessian·v` matches central FD of the analytic gradient to relative `1e-9`, at a random `γ≠0` config and a relaxed jammed state; symmetry to `1e-12`.
- Backbone DOS Hessian is PSD with exactly two ≈0 eigenvalues; `ω_max` finite.

`test_moduli.py`
- `B>0` on null-protocol jammed states; `G ≥ -1e-8` (**shear-stabilized by construction**).
- `B,G` cross-checked against a direct affine-deformation finite-difference of the relaxed enthalpy at small `N` (apply a small `δℓ`/`δγ`, re-minimize the complementary dofs, fit curvature).

`test_rollout.py`
- `run_episodes_batch` parity: a batch episode equals the Python `step`-by-`step` loop for the same seed (bitwise).
- Episode-parallel reproducibility: shuffling the seed order does not change per-seed results.

`test_learner.py`
- PPO mean reward improves over a handful of updates on a tiny config (`N=32`, short `T_cap`), deterministically.
- CEM elite mean is non-decreasing.
- Checkpoint round-trip: save → load → identical policy outputs; resume produces the same next-round policy as an uninterrupted run.

---

## 9. Implementation phases (build order)

Build incrementally; each phase ends at its §8 gate. This mirrors the "validate the numerical engine before assembling anything on top" discipline.

- [ ] **Phase 0 — Scaffold & build.** Repo layout (§1), CMake + pybind11 + Eigen wired, `jamrl._core` imports, `pip install -e .` works, a trivial bound function returns. Conda/venv env reproducible.
- [ ] **Phase 1 — Types & system.** `System`, reduced coords, radii init (50:50 `0.5/0.7`), RNG (splitmix64 + a stream PRNG), `make_system`. 
- [ ] **Phase 2 — `evaluate` (O(N²)).** Energy/grad/biased enthalpy (§3.2), `γ`-wrap (§3.3). **Gate:** `test_evaluate.py`.
- [ ] **Phase 3 — L-BFGS & env.** Minimizer with all §4 fixes; the MDP `step`, reward, termination, finish-and-measure, outcome taxonomy; null-φ helper. **Gate:** `test_lbfgs.py`, `test_env.py` (at `N=32`).
- [ ] **Phase 4 — Cell lists & OpenMP.** Lees-Edwards linked-cell + Verlet skin; episode-parallel batch path; thread controls. **Gate:** `test_cells.py` (correctness to `1e-12` + scaling to `N=1024`).
- [ ] **Phase 5 — Hessian, moduli, DOS.** Sparse full Hessian, backbone DOS Hessian, Schur moduli, dense + Spectra eig. **Gate:** `test_hessian.py`, `test_moduli.py`.
- [ ] **Phase 6 — Policy MLP & batch runner.** Embedded tanh MLP + Gaussian sampling; `run_episodes_batch` returning trajectories + jammed-state data + optional Hessian/spectrum. **Gate:** `test_rollout.py`.
- [ ] **Phase 7 — Python orchestration.** `config.py` (defaults+YAML+argparse), `seeding.py`, `storage.py` (npz/h5/parquet, §7), `rollout.py`. **Gate:** `jamrl run-local --rounds 2 --workers 2 --episodes-per-worker 2 --N 64` completes and writes valid data.
- [ ] **Phase 8 — Learner.** `policy.py`, `ppo.py`, `cem.py`, checkpoint/resume, eval + metrics → parquet. **Gate:** `test_learner.py`; `run-local` end-to-end shows reward improving on a small `N`.
- [ ] **Phase 9 — SLURM.** `slurm.py`, sbatch templates, `submit` self-perpetuating chain, `afterany` + robust aggregation, `--resume`, `STOP`/`DONE`, `status`. **Gate:** `sbatch --test-only` dry run wires dependencies correctly; a small real run (`--N 256 --rounds 3 --workers 4`) completes and perpetuates.
- [ ] **Phase 10 — Post-processing & analysis.** `postprocess.py` (sharded diagonalization → spectra/moduli → parquet) as a non-blocking dependency; `plot`/`analyze`; `compact`. **Gate:** spectra and moduli for a completed round land in parquet and plot; training is verified not to stall waiting on it.

---

## 10. Parameter defaults reference

Strong defaults; everything overridable via `argparse` (and YAML). Ranges are guidance, not hard limits.

| Group | Flag | Default | Range / notes |
|---|---|---|---|
| System | `--N` | `1024` | 32–8192; first target 1024 |
| | `--P` | `1e-3` | 1e-5 – 1e-1 (target pressure) |
| | `--phi0` | `0.80` | initial packing fraction |
| Loading | `--kappa-P` | `1.0` | compression authority |
| | `--kappa-sigma` | `0.5` | shear authority |
| | `--n-relax` | `20` | L-BFGS iters per macro-step (≥1) |
| | `--T-cap` | `60` | macro-steps per episode |
| Reward | `--w-phi` | `400` | density reward weight |
| | `--c-step` | `0.01` | per-step cost |
| | `--fail-pen` | `2.0` | failure penalty |
| | `--trunc-pen` | `0.5` | finish-and-measure penalty |
| | `--quiesce-tol` / `--quiesce-n` | `0.05` / `3` | early-release detection |
| | `--finish-cap` | `12000` | floor of pressure-scaled finish budget |
| Tolerances | `--ftol-abs` / `--ftol-rel-P` / `--ptol` | `1e-10` / `1e-5` / `1e-4` | convergence |
| Learner | `--algo` | `ppo` | `ppo` \| `cem` |
| | `--lr` | `3e-4` | PPO Adam lr |
| | `--gamma` / `--lam` | `0.995` / `0.95` | discount / GAE |
| | `--clip` | `0.2` | PPO clip |
| | `--ppo-epochs` / `--minibatch` | `6` / `1024` | |
| | `--ent-coef` / `--vf-coef` | `3e-3` / `0.5` | |
| | `--logstd-init` | `-0.5` | initial policy σ ≈ 0.61 |
| | `--hidden` | `64,64` | MLP widths |
| | `--cem-pop` / `--cem-elite-frac` | `64` / `0.25` | |
| | `--cem-sigma0` / `--cem-eps-per-cand` | `0.3` / `4` | |
| Campaign | `--rounds` | `1000` | training rounds |
| | `--workers` | `64` | SLURM array size / round |
| | `--episodes-per-worker` | `8` | |
| | `--eval-seeds` | `101..106` | held-out |
| | `--parallel-mode` | `episode` | `episode` (deterministic) \| `intra` |
| | `--threads-per-task` | `16` | cpus-per-task |
| Data | `--save-hessian` | `sparse` | `none`\|`spectrum`\|`sparse`\|`dense` — use `spectrum` for long runs |
| | `--hessian-stride` | `1` | save every k-th state's Hessian |
| SLURM | `--partition` / `--account` | `""` | cluster-specific |
| | `--time-rollout` / `--time-learn` / `--time-post` | `02:00:00` / `00:20:00` / `01:00:00` | |
| | `--mem-rollout` / `--mem-learn` / `--mem-post` | `8G` / `8G` / `16G` | |
| | `--dependency-mode` | `afterany` | `afterany` (robust) \| `afterok` (strict) |
| | `--min-worker-frac` | `0.6` | learner aborts below this |
| Misc | `--seed` | `12345` | campaign seed |
| | `--campaign-root` / `--name` | `./campaigns` / `run` | |

---

## 11. Run recipes

**Local debug (no SLURM), tiny and fast:**
```bash
jamrl run-local --N 64 --rounds 3 --workers 2 --episodes-per-worker 2 \
                --T-cap 30 --name smoke
jamrl status --campaign campaigns/smoke
```

**Single-node, many episodes, one round (sanity at scale):**
```bash
jamrl run-local --N 1024 --rounds 1 --workers 1 \
                --episodes-per-worker 32 --threads-per-task 32 --name scale1k
```

**Full cluster campaign (one submission perpetuates all rounds):**
```bash
jamrl submit --N 1024 --rounds 1000 --workers 64 --episodes-per-worker 8 \
             --threads-per-task 16 --partition cpu --account myproj \
             --time-rollout 02:00:00 --save-hessian spectrum --name big1k
# inspect the wired dependency chain without queueing:
jamrl submit --N 1024 --name big1k --test-only
```

**Resume after interruption / stop cleanly:**
```bash
jamrl submit --resume --name big1k          # continue from last completed round
touch campaigns/big1k/STOP                   # next learner stops instead of resubmitting
```

**Analysis:**
```bash
jamrl plot --campaign campaigns/big1k        # reward curve, Δφ, loading magnitudes, DOS evolution
jamrl compact --campaign campaigns/big1k --keep-spectrum --round-range 0:500
```

A YAML config can replace the flags entirely: `jamrl submit --config big1k.yaml` (CLI flags still override YAML).

---

## 12. Optional / stretch features

- **Low-frequency DOS at large `N`:** `save_hessian=spectrum` with Spectra shift-invert (`--dos-k 60`, small negative shift to avoid the singular `σ=0`) instead of dense LAPACK above `N≈2048`; track `ω*` (the plateau edge) vs `Δz` as a phenotype observable.
- **Transfer across `N`:** because observations are intensive, a policy trained at `N=1024` can be evaluated at `N=2048/4096` with no retraining — add a `jamrl eval --policy ... --N 4096` path and report generalization.
- **GPU learner:** `--device cuda`; negligible benefit for these MLP sizes but trivial to support.
- **Domain randomization:** randomize `P` (and optionally `φ0`) per episode within a range to learn pressure-robust protocols; add `--P-range`.
- **Policy averaging / EMA** of weights across rounds for smoother eval curves.
- **Experiment tracking:** optional Weights & Biases / TensorBoard hooks in `learn.py` (guarded by `--wandb`).
- **Reward shaping toggles:** expose a `--reward-mode` to add (small) shear-stability or coordination bonuses on top of the density term, for ablations — keep density-primary as the default so results stay comparable to the prototype.

---

### Final notes to the implementer

The single most important rule: **the engine is validated by the analytic gate (§8), not by inspection.** Implement the `O(N²)` `evaluate` and the L-BFGS hazard fixes first, make `test_evaluate`/`test_lbfgs`/`test_env` pass at `N=32`, and only then add cell-lists/OpenMP and require the fast path to reproduce the slow path bit-for-bit. The four numerical hazards in §4 are not optional hardening — each one corresponds to a concrete divergence already observed (sub-precision line-search spin, `L~10¹⁴³` box runaway, joint `γ/ℓ` blow-up, low-`P` under-jamming) and each has a test. Once the core is trustworthy, the SLURM layer is just plumbing: one `jamrl submit` launches a rollout array, a dependent learner that updates the agent and resubmits the next round, and a non-blocking post-processing job that turns saved Hessians into DOS spectra and moduli.
