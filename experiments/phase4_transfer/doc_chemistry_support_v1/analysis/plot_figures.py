"""Render the fixed chemical-support comparison tables; no fits or selection."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
curves = pd.read_csv(ROOT / "k_curves.csv")
effects = pd.read_csv(ROOT / "comparisons.csv")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
styles = {
    "legacy": ("#737B86", "o", "--", "Legacy recurrent states"),
    "masks_aug": ("#BC9654", "s", ":", "+ Mask-state coordinates"),
    "chemistry_aug": ("#176BA0", "^", "-", "+ Chemistry-state coordinates"),
    "selected": ("#A44169", "D", "-.", "Validation-selected basis"),
}
pipelines = [
    ("neural_chemistry", "Neural correction"),
    ("neural_chemistry_integrated", "Neural + ecological integration"),
    ("tree_chemistry", "Chemistry tree"),
]
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), sharey=True)
for i, ((pipe, title), ax) in enumerate(zip(pipelines, axes)):
    for variant, (color, marker, line, label) in styles.items():
        group = curves[curves.model_name.eq(f"{pipe}_{variant}")].sort_values("k")
        ax.plot(
            group.k,
            group.mae,
            color=color,
            marker=marker,
            linestyle=line,
            markersize=4,
            linewidth=1.25,
            label=label,
            markerfacecolor="white" if variant == "masks_aug" else color,
        )
    ax.set_title(title, pad=12)
    ax.text(
        -0.1, 1.07, chr(97 + i), transform=ax.transAxes, fontweight="bold", fontsize=12
    )
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Target DOC support observations, K")
    ax.set_ylim(1.50, 1.83)
    ax.grid(axis="y", linewidth=0.5, color="#E4E6E9")
    ax.set_axisbelow(True)
axes[0].set_ylabel("DOC MAE (mg L$^{-1}$)")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.05),
    frameon=False,
    columnspacing=2.2,
    handlelength=2.5,
)
fig.text(
    0.5,
    0.015,
    "Same fixed queries; seed means within each partition, then equal partition means. K0 and K1 are exactly unchanged.",
    ha="center",
    fontsize=8,
    color="#555D67",
)
fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.29, wspace=0.17)
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"chemical_support_k_curves.{ext}", dpi=220)
plt.close(fig)

rows = []
for pipe, title in [
    ("neural_chemistry", "Neural"),
    ("neural_chemistry_integrated", "Neural + ecology"),
    ("tree_chemistry", "Tree"),
]:
    for candidate, ref, label in [
        ("chemistry_aug", "legacy", "Chemistry − legacy"),
        ("chemistry_aug", "masks_aug", "Chemistry − masks"),
        ("selected", "legacy", "Selected − legacy"),
    ]:
        rows.append(
            (
                f"{pipe}_{candidate}_vs_{ref}_k",
                f"{title}: {label}",
                styles[candidate][0],
            )
        )
rows += [
    (
        "selected_neural_integrated_vs_point_integrated_legacy_k",
        "Selected neural + ecology − general reference",
        "#333F4B",
    ),
    (
        "selected_neural_integrated_vs_tree_chemistry_selected_k",
        "Selected neural + ecology − selected tree",
        "#333F4B",
    ),
]
fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.3), sharex=True, sharey=True)
for k, ax in zip((3, 5), axes):
    for i, (prefix, label, color) in enumerate(rows):
        value = effects[
            effects.comparison.eq(f"{prefix}{k}")
            & effects.metric.eq("mae")
            & effects.region.eq("overall")
        ].iloc[0]
        ax.errorbar(
            value.delta_value,
            i,
            xerr=np.array(
                [
                    [value.delta_value - value.delta_ci_low],
                    [value.delta_ci_high - value.delta_value],
                ]
            ),
            fmt="o" if k == 3 else "D",
            markersize=4.5,
            color=color,
            elinewidth=1.3,
            capsize=2.5,
            capthick=0.8,
        )
    ax.axvline(0, color="#5C6672", linewidth=0.8, linestyle="--")
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color="#E4E6E9", linewidth=0.6)
    ax.set_title(f"{k} support observations", pad=14)
    ax.set_xlabel("Paired ΔMAE (mg L$^{-1}$)")
    ax.set_xlim(-0.055, 0.075)
    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04, 0.06])
    ax.grid(axis="x", color="#EFF0F2", linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(rows)), [r[1] for r in rows])
axes[0].invert_yaxis()
axes[0].text(
    -0.06, 1.08, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=12
)
axes[1].text(
    -0.06, 1.08, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=12
)
fig.text(
    0.63,
    0.09,
    "← Lower candidate error                         Higher candidate error →",
    ha="center",
    fontsize=8,
    color="#555D67",
)
fig.text(
    0.63,
    0.035,
    "95% intervals: 5,000 paired whole-station bootstrap draws; 172 unique stations, 10,520 unique query cells.\nAll 22 fixed contrasts are shown. Development station partitions; no target-based model selection.",
    ha="center",
    fontsize=8,
    color="#555D67",
    linespacing=1.5,
)
fig.subplots_adjust(left=0.37, right=0.99, top=0.87, bottom=0.20, wspace=0.15)
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"chemical_support_paired_effects.{ext}", dpi=220)
plt.close(fig)
print("Rendered two figures, PNG + vector PDF, from bound analysis tables.")
