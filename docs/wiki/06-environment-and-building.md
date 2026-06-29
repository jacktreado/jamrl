# 06 — Environment & building `_core`

> ⚙️ **Always use the `jamrl` conda environment for every `jamrl` command** —
> build, test, CLI, and scripts. It is the only supported environment for this
> repo. **Activate it first**, in every shell, before anything else:
> ```bash
> conda activate jamrl
> ```

## Activate the environment

The `jamrl` env (from [`environment.yml`](../../environment.yml)) carries the
full stack — Python 3.11, conda-forge PyTorch (single OpenMP runtime), NumPy,
h5py, and the build toolchain. Create it once, then activate it in **every**
shell where you run `jamrl`, build `_core`, or run the tests:

```bash
# one-time creation
conda env create -f environment.yml

# every shell, before any jamrl command
conda activate jamrl
```

Confirm you are in the right place before running anything:

```bash
which jamrl        # → .../envs/jamrl/bin/jamrl
python -c "import jamrl._core"   # no output = the C++ core imports cleanly
```

If `which jamrl` does **not** point inside `envs/jamrl`, you have not activated
the environment — stop and `conda activate jamrl`.

## Building / rebuilding the C++ core (`_core`)

The heavy physics lives in the C++ extension `jamrl._core` (sources under
[`cpp/`](../../cpp/)). **Any time the C++ source changes** — pulling updates,
editing `cpp/`, or changing reward logic (the reward is computed in C++) — you
must rebuild `_core`, or you will silently run stale physics.

> Activate the env first (`conda activate jamrl`); all commands below assume it.

### Canonical: editable install (recommended)

```bash
conda activate jamrl
pip install -e . --no-build-isolation
```

This scikit-build-core editable install rebuilds `_core` against the env's
Python 3.11 and points the import at the fresh build. It also registers the
`jamrl` console script. Use this on your laptop **and** on the cluster.

### Fast iteration: `scripts/build.sh`

For quick C++ edit/rebuild cycles, this drops a freshly built `_core*.so` next to
the package in `src/jamrl/` (so `PYTHONPATH=src pytest` picks it up):

```bash
conda activate jamrl
PYTHON=$(which python3) bash scripts/build.sh
```

Passing `PYTHON=$(which python3)` (with the env active) ensures CMake builds
against the `jamrl` interpreter and links the env's own libomp — a single
in-process OpenMP runtime. On macOS, linking a second OpenMP runtime (e.g.
Homebrew's libomp) triggers "OMP Error #15" once NumPy/torch load; see
[Recipes ▸ FAQ](05-recipes.md#faq).

### Verify the build

```bash
conda activate jamrl
python -c "import jamrl._core as c; print('OpenMP:', c.has_openmp())"
pytest -q          # run the validation gates
```

## When do I need to rebuild?

| You did this | Rebuild `_core`? |
|---|---|
| Pulled changes touching `cpp/` | **Yes** |
| Changed a reward mode / reward weights logic in C++ | **Yes** (reward is computed in `_core`) |
| Edited only Python under `src/jamrl/` | No (editable install already reflects it) |
| Edited only docs / configs / notebooks | No |

← Back to [wiki home](README.md)
