"""Plot fitted and station-CV kernel deltas from source-only summary tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parent
fitted = pd.read_csv(root / "full_validation_comparisons.csv")
cv = pd.read_csv(root / "incremental_cv_comparisons.csv")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
pipelines = [
    ("neural_chemistry", "Direct neural"),
    ("neural_chemistry_integrated", "Neural + ecology"),
    ("tree_chemistry", "Chemistry tree"),
]
for k, ax in zip((3, 5), axes):
    for i, (pipeline, label) in enumerate(pipelines):
        name = f"{pipeline}_chemistry_vs_parent_k{k}"
        a = (
            fitted[fitted.comparison.eq(name) & fitted.region.eq("overall")]
            .iloc[0]
            .delta_mae
        )
        b = cv[cv.comparison.eq(name) & cv.region.eq("overall")].iloc[0].delta_mae
        ax.plot([a, b], [i, i], color="#CDD2D8", linewidth=1.2, zorder=1)
        ax.scatter(
            a,
            i,
            s=36,
            marker="o",
            color="#BC9654",
            zorder=3,
            label="Full-validation selected score" if i == 0 else None,
        )
        ax.scatter(
            b,
            i,
            s=36,
            marker="D",
            color="#176BA0",
            zorder=3,
            label="Held-station incremental CV" if i == 0 else None,
        )
    ax.axvline(0, color="#65717F", linewidth=0.8, linestyle="--")
    ax.set_title(f"K = {k} DOC observations", fontweight="bold", pad=13)
    ax.set_xlabel("Kernel − parent ΔMAE (mg L$^{-1}$)")
    ax.set_xlim(-0.008, 0.0065)
    ax.set_xticks([-0.006, -0.003, 0, 0.003, 0.006])
    ax.grid(axis="x", color="#EEF0F2", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(2.4, -0.4)
axes[0].set_yticks(range(3), [label for _, label in pipelines])
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.53, 0.13),
    frameon=False,
    ncol=2,
    fontsize=9,
)
fig.text(
    0.53,
    0.035,
    "Source-validation data only. Parent already selected; CV reselects kernel eta/bandwidth on other station folds.\nNegative values favor the kernel. These development scores are not outer-test confirmation; no confidence intervals are shown.",
    ha="center",
    fontsize=8,
    color="#59616B",
    linespacing=1.5,
)
fig.subplots_adjust(left=0.19, right=0.985, top=0.80, bottom=0.35, wspace=0.17)
for extension in ("png", "pdf"):
    fig.savefig(root / f"kernel_fitted_vs_station_cv.{extension}", dpi=220)
plt.close(fig)
