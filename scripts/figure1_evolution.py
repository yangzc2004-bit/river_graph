"""Figure 1 (data-accurate): method evolution ladder from the frozen tables.

G0 from frozen benchmark.csv; H1/H2/H2X from the 5-seed multiseed table.
Output: experiments/figures/figure1_method_evolution_data.png

Usage: python scripts/figure1_evolution.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("experiments/figures")
MODELS = ["G0", "H1", "H2", "H2X"]
LABELS = ["G0\nriver graph", "H1\n+ direction", "H2\n+ transport gates",
          "H2X\n+ ecological encoder"]


def r2_of(df, model, masks):
    sub = df[(df["model"] == model) & (df["mask"].isin(masks))]
    return sub["r2"].mean(), sub["r2"].std()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv("experiments/frozen_results/benchmark.csv")
    multi = pd.read_csv("experiments/frozen_results/benchmark_multiseed.csv")

    e1 = [f"e1_r20_seed{s}" for s in (42, 43, 44)]
    e3 = [f"e3_spatial_seed{s}" for s in (42, 43, 44)]
    series = {"E1 random mask (r20)": e1, "E2b future reconstruction": ["e2b_partial"],
              "E3 unseen stations": e3}

    data = {}
    g0 = frozen[frozen["model"] == "G0_gcn_river"]
    data["G0"] = {name: (g0[g0["mask"].isin(ms)]["r2"].mean(), 0.0)
                  for name, ms in series.items()}
    for m in ["H1", "H2", "H2X"]:
        sub = multi[multi["model"] == m]
        data[m] = {name: (sub[sub["mask"].isin(ms)]["r2"].mean(),
                          sub[sub["mask"].isin(ms)]["r2"].std())
                   for name, ms in series.items()}

    colors = {"E1 random mask (r20)": "#1f4e79", "E2b future reconstruction": "#2e75b6",
              "E3 unseen stations": "#2e8b57"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(MODELS))
    for name, color in colors.items():
        ys = [data[m][name][0] for m in MODELS]
        ax.plot(xs, ys, "o-", color=color, lw=2, ms=7, label=name)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=color)
    ax.set_xticks(xs, LABELS)
    ax.set_ylabel("R² (higher is better)")
    ax.set_ylim(0, 0.6)
    ax.set_title("Figure 1 | Stepwise model evolution on the DOC benchmark\n"
                 "(G0: frozen single seed; H1/H2/H2X: 5-seed mean)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "figure1_method_evolution_data.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    for name in series:
        print(name, [f"{data[m][name][0]:.3f}" for m in MODELS])


if __name__ == "__main__":
    main()
