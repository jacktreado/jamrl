# 04 — Analysis

> ⚙️ **Activate the `jamrl` conda env first** (`conda activate jamrl`) — run
> `jamrl analyze`/`eval`/`compact` and the notebooks in that environment. See
> [06 — Environment & building](06-environment-and-building.md).

How to get from a campaign (running or finished) to plots on your laptop, plus
the two policy-evaluation/maintenance commands.

## `jamrl analyze`

Condenses a campaign into a single portable HDF5 for offline notebooks:

```bash
jamrl analyze --campaign /home/data/$USER/campaigns/big1k
# → writes /home/data/$USER/campaigns/big1k/analysis/campaign_analysis.h5
```

**It works mid-campaign.** `analyze` reads only the rounds that have completed
data and skips the rest, so you can run it any time — at round 500 of a 2000-round
campaign, you get an h5 covering everything done so far. Re-run it later to pick
up new rounds; it overwrites the h5.

Strides keep the file small (full per-round trajectories and spectra are large):

```bash
jamrl analyze --campaign <camp> --spectra-stride 5 --traj-stride 10 --out /tmp/big1k.h5
```

- `--spectra-stride` (default 10) — sample VDOS/mechanics every N rounds.
- `--traj-stride` (default 25) — sample trajectory/observation/action data every N rounds.
- `--out` — output path (default `<camp>/analysis/campaign_analysis.h5`).

The h5 contains groups for `summary`, `policy`, `vdos`, `mechanics`, optional
`vdos_box`/`projections`, plus `actions`, `observations`, and raw `trajectories`
(see `build_campaign_analysis` in
[`src/jamrl/campaign_analysis.py`](../../src/jamrl/campaign_analysis.py)).

## Transfer to your laptop and open the notebooks

```bash
scp user@cluster:/home/data/$USER/campaigns/big1k/analysis/campaign_analysis.h5 .
```

Then open the notebooks in [`notebooks/`](../../notebooks/). Each reads the h5;
**edit the path cell at the top** to point at your local copy (paths are
hardcoded).

| notebook | shows |
|---|---|
| [`01_training_curves.ipynb`](../../notebooks/01_training_curves.ipynb) | reward, objective (Δφ or ΔG), success, σ, B/G/Δz, rattlers, ω*, wall time |
| [`02_policy_evolution.ipynb`](../../notebooks/02_policy_evolution.ipynb) | weight-norm and layer-weight evolution over rounds |
| [`03_vdos_evolution.ipynb`](../../notebooks/03_vdos_evolution.ipynb) | vibrational density of states + ω* over rounds |
| [`04_mechanical_distributions.ipynb`](../../notebooks/04_mechanical_distributions.ipynb) | B, G, Δz, φ distributions and their spread |
| [`05_action_obs_distributions.ipynb`](../../notebooks/05_action_obs_distributions.ipynb) | action (a_P, a_S) and observation histograms over rounds |
| [`06_relaxation_mode_projection.ipynb`](../../notebooks/06_relaxation_mode_projection.ipynb) | terminal-motion projection onto relaxation modes, soft-mode fraction |

## Re-evaluate at a different system size

Transfer a trained policy to a larger N and run a deterministic eval:

```bash
jamrl eval --campaign <camp> --round 500 --N 4096
```

Loads `policy/round_0500.npz`, runs a greedy rollout at the given `--N` (0 = the
campaign's N), and prints the eval stats.

## Reclaim disk: `compact`

Down-convert old per-state Hessians to compact spectra:

```bash
jamrl compact --campaign <camp> --round-range 0:400          # compact rounds 0–399
jamrl compact --campaign <camp> --round-range 0:400 --keep-spectrum
```

Omit `--round-range` to compact all rounds.

→ Next: [05 — Recipes](05-recipes.md)
