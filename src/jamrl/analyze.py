"""Status reporting and plotting from the summary parquet (plan 5.3 / 10)."""
from __future__ import annotations

import os

from jamrl import storage
from jamrl.config import Config


def _load_cfg(camp):
    p = os.path.join(camp, "config.yaml")
    return Config.from_yaml(p) if os.path.exists(p) else None


def print_status(camp: str) -> None:
    cfg = _load_cfg(camp)
    df = storage.read_summary(camp)
    done = os.path.exists(os.path.join(camp, "DONE"))
    stop = os.path.exists(os.path.join(camp, "STOP"))

    print(f"campaign: {camp}")
    if cfg is not None:
        print(f"  algo={cfg.algo}  N={cfg.N}  P={cfg.P}  rounds_target={cfg.rounds}")
    print(f"  rounds_completed={len(df)}  DONE={done}  STOP={stop}")

    # The objective column tracked depends on the reward mode.
    mode = getattr(cfg, "reward_mode", "density") if cfg is not None else "density"
    obj_col, obj_lbl = ("eval_dG", "eval_dG") if mode == "shear_modulus" else ("eval_dphi", "eval_dphi")

    if len(df):
        tail = df.tail(8)
        print(f"  recent rounds (round | mean_reward | {obj_lbl} | success | sigma):")
        for _, row in tail.iterrows():
            print(f"    {int(row['round']):5d} | {row['mean_reward']:+9.3f} | "
                  f"{row.get(obj_col, float('nan')):+9.4f} | {row['eval_success']:.2f} | "
                  f"{row.get('sigma_policy', float('nan')):.3f}")
        last = df.iloc[-1]
        print("  last eval:")
        for k in ("Bbar", "Gbar", "dzbar", "rattler_frac", "shear_stable_frac",
                  "mean_absaP", "mean_absaS", "mean_absgamma"):
            if k in last:
                print(f"    {k:18s} = {last[k]}")

    # last submitted round's job ids
    rdir = os.path.join(camp, "rounds")
    if os.path.isdir(rdir):
        files = sorted(f for f in os.listdir(rdir) if f.startswith("round_"))
        if files:
            r = int(files[-1].split("_")[1].split(".")[0])
            rj = storage.read_round_json(camp, r)
            print(f"  last round json: r={r} jobs={{roll:{rj.get('roll_jid')}, "
                  f"learn:{rj.get('learn_jid')}, post:{rj.get('post_jid')}}}")


def plot_campaign(camp: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = _load_cfg(camp)
    df = storage.read_summary(camp)
    outdir = os.path.join(camp, "analysis", "plots")
    os.makedirs(outdir, exist_ok=True)
    if df.empty:
        print("[plot] no summary data yet")
        return []

    # Primary objective panel tracks the active reward mode.
    mode = getattr(cfg, "reward_mode", "density") if cfg is not None else "density"
    obj = ("eval_dG", "eval ⟨G − G_null⟩") if mode == "shear_modulus" else ("eval_dphi", "eval ⟨φ − φ_null⟩")

    x = df["round"].to_numpy()
    panels = [
        ("mean_reward", "training mean reward"),
        obj,
        ("eval_success", "eval success rate"),
        ("Gbar", "mean shear modulus G"),
        ("dzbar", "mean Δz"),
        ("mean_absaP", "mean |a_P|"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, title) in zip(axes.ravel(), panels):
        if col in df:
            ax.plot(x, df[col].to_numpy(), ".-")
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    fig.suptitle(f"jamrl campaign: {os.path.basename(camp)}")
    fig.tight_layout()
    out = os.path.join(outdir, "summary.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return [out]
