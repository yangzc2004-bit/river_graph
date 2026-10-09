"""Plot precomputed K0 reconstruction strata with unchanged source thresholds."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

GEO = Path("experiments/phase4_transfer/doc_geographical_confirmation_v1/analysis_5seed_final/strata_summary.csv")
EXTERNAL = Path("experiments/phase4_transfer/doc_external_replication_v1/analysis/strata.csv")
OUTPUT = Path("docs/paper/latex/figures")
MODELS = {"current_model": ("#737E87", "Preceding full model"),
          "station_hidden_trees": ("#2F7F71", "Station-hidden trees"),
          "unmonitored_integrated": ("#CE7B2E", "Fixed upgrade")}
STRATA = (("hydro_channels_available", "Available hydro channels", ("None", "One", "Both")),
          ("ecological_novelty", "Ecological novelty", ("Low", "Middle", "High")),
          ("source_similarity_distance", "Nearest-source distance", ("Low", "Middle", "High")))


def main():
    geo, external = pd.read_csv(GEO), pd.read_csv(EXTERNAL)
    geo = geo[geo.k.eq(0)].copy()
    external["stratum"] = external.stratum.str.replace("_group", "", regex=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8))
    for row, table in enumerate((geo, external)):
        for column, (stratum, title, labels) in enumerate(STRATA):
            ax = axes[row, column]
            for name, (color, label) in MODELS.items():
                part = table[table.model_name.eq(name) & table.stratum.eq(stratum)].set_index("group")
                if not part.index.is_unique:
                    raise ValueError("stratum summaries contain duplicate groups")
                values = part.mae.reindex(range(3)).to_numpy()
                ax.plot(range(3), values, "o-", color=color, label=label, linewidth=1.5, markersize=6)
            count = table[table.model_name.eq("station_hidden_trees") & table.stratum.eq(stratum)].set_index("group")
            tick_labels = []
            for group in range(3):
                if group not in count.index:
                    note = "empty"
                elif row == 0:
                    note = f"{int(count.loc[group, 'n_regions'])} regions"
                else:
                    note = f"{int(count.loc[group, 'n_stations'])} sites"
                tick_labels.append(f"{labels[group]}\n{note}")
            finite = table[table.stratum.eq(stratum) & table.model_name.isin(MODELS)].mae.to_numpy()
            ax.set_ylim(0, np.max(finite)*1.23)
            ax.set_xticks(range(3), tick_labels, fontsize=9)
            ax.set(title=title, ylabel=r"K0 MAE (mg L$^{-1}$)")
            ax.grid(axis="y", color="#E7ECEF", linewidth=.6)
            ax.set_axisbelow(True)
            ax.text(-.15, 1.03, "abcdef"[row*3+column], transform=ax.transAxes,
                fontsize=15, fontweight="bold")
    fig.suptitle("DOC reconstruction across environmental strata", fontsize=17, fontweight="bold", y=.98)
    fig.text(.045, .865, "Geographical holdouts", fontsize=12, fontweight="bold", color="#414B53")
    fig.text(.045, .465, "Independent basin", fontsize=12, fontweight="bold", color="#414B53")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055), frameon=False, ncol=3)
    fig.text(.5, .031, "Geography: seed means then equal usable regions. External: fixed five-seed prediction, cell-weighted error.",
        ha="center", fontsize=9, color="#65747E")
    fig.text(.5, .009, "Ecological and distance groups use saved source tertiles; empty target groups remain visible.",
        ha="center", fontsize=8.7, color="#65747E")
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.16, hspace=.65, wspace=.37)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT/f"doc_transfer_conditions_v1.{extension}", dpi=240)
    plt.close(fig)
    (OUTPUT/"doc_transfer_conditions_v1_sources.json").write_text(json.dumps({
        "geographical_strata": {"path": str(GEO), "sha256": sha256_file(GEO)},
        "external_strata": {"path": str(EXTERNAL), "sha256": sha256_file(EXTERNAL)},
        "plotter_sha256": sha256_file(__file__), "interpretation": "descriptive, fixed source-defined strata"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
