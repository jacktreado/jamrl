# jamrl — distributed RL of box-control jamming protocols

A reinforcement-learning lab for jamming: an agent applies compression/shear
loads to a periodic Lees-Edwards shear box while an L-BFGS minimizer relaxes a
2D bidisperse harmonic-disk packing, rewarded for producing **denser jammed
states than a zero-action null protocol**. The heavy physics runs in a C++ core
(`jamrl._core`, OpenMP); a Python layer handles config, rollouts, the learner,
SLURM orchestration, and post-processing.

Built phase-by-phase from `docs/plans/jamrl_hpc_implementation_plan.md`; each
phase is gated by a `pytest` suite (the engine is validated by the analytic
gate, not by inspection). See `docs/physics.md` for the formulas and the
numerical-hazard fixes.

## Architecture

```
policy(r) → ROLLOUT ARRAY (W workers × E episodes)  →  trajectories + jammed states
                          │ afterany
              ┌───────────┴───────────┐
              ▼                        ▼
        LEARN (PPO/CEM)         POSTPROCESS (DOS, moduli)
        → policy(r+1)           → spectra/parquet  (non-blocking)
        → submits round r+1
```

One `jamrl submit` launches a self-perpetuating campaign: each learner job
updates the agent, writes the next policy, and submits the next round.

## Install / build

Dependencies are fetched by CMake (Eigen, pybind11, Spectra); OpenMP is
compiler-provided. Python deps are in `pyproject.toml` / `environment.yml`.

```bash
# editable install (rebuilds _core on demand)
pip install -e . --no-build-isolation

# or fast local C++ iteration (drops _core*.so into src/jamrl/)
PYTHON=$(which python3) bash scripts/build.sh

# run the validation gates
PYTHONPATH=src pytest -q          # (or just `pytest` after install)
```

On macOS the build links the **Python environment's** libomp so a single
OpenMP runtime lives in-process (linking a second one — e.g. Homebrew's —
triggers "OMP Error #15" once numpy/numba are imported).

### Learner backend

The PPO learner has two interchangeable backends, selected by
`--backend {auto,torch,numpy}` (default `auto`):

- **torch** — the plan's PyTorch path (autograd, `.pt` checkpoints). Default
  whenever torch is importable. Use the conda env from `environment.yml`, where
  torch comes from **conda-forge** (a single OpenMP runtime, no conflict).
- **numpy** — a self-contained backend (manual backprop + Adam) for environments
  where torch is unavailable or broken (e.g. **pip**-installed torch segfaults
  inside some anaconda envs due to a bundled-libiomp5 clash).

Both export the same actor-facing policy `npz` consumed by the C++ runner, so the
engine and the (gradient-free) CEM path are identical across backends.

## Run recipes

```bash
# local debug, no SLURM
jamrl run-local --N 64 --rounds 3 --workers 2 --episodes-per-worker 2 \
                --T-cap 30 --name smoke
jamrl status --campaign campaigns/smoke
jamrl plot   --campaign campaigns/smoke

# full cluster campaign (one submission perpetuates all rounds)
jamrl submit --N 1024 --rounds 1000 --workers 64 --episodes-per-worker 8 \
             --threads-per-task 16 --partition cpu --account myproj \
             --save-hessian spectrum --name big1k
jamrl submit --N 1024 --name big1k --test-only     # inspect wiring, don't queue

# resume / stop
jamrl submit --resume --name big1k
touch campaigns/big1k/STOP                          # next learner stops cleanly

# transfer a trained policy to a larger N (intensive observations)
jamrl eval --campaign campaigns/big1k --round 500 --N 4096
```

A YAML config can replace flags: `jamrl submit --config big1k.yaml` (CLI flags
still override YAML).

## Layout

`cpp/` C++ core (`include/jamcore`, `src`, `bindings`) · `src/jamrl/` Python
package · `tests/` per-phase gates · `docs/` plan + physics reference.
