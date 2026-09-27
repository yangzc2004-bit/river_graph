"""Render manuscript figures with descriptive scenario names from frozen results.

Run from the repository root; does not train or modify experiment products.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("experiments/phase4_transfer/temporal_h2x_v1")
OUT = Path("docs/paper/latex/figures")
ANALYTES = ["doc", "ph", "spec_conductance"]
ANALYTE_LABELS = {"doc": "DOC", "ph": "pH", "spec_conductance": "Specific conductance"}
MASKS = ["e1_r20_seed42", "e2a_strict", "e2b_partial", "e3_spatial_seed42"]
SCENARIOS = {
    "e1_r20_seed42": "Random gaps",
    "e2a_strict": "Unobserved-period extrapolation",
    "e2b_partial": "Observation-assisted extrapolation",
    "e3_spatial_seed42": "Unmonitored stations",
}


def make_figures(product: pd.DataFrame, station_error: pd.DataFrame) -> None:
    summary = pd.read_csv(ROOT / 't8_synthesis' / 'main_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, analyte in zip(axes, ANALYTES):
        data = summary[summary.analyte.eq(analyte)].copy()
        pivot = data.set_index('mask')['reduction_pct'].reindex(MASKS)
        matrix = pivot.to_numpy()[None, :]
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-5, vmax=60, aspect='auto')
        ax.set_xticks(
            range(4),
            ['Random gaps', 'Strict temporal\nextrapolation',
             'Context-assisted\nextrapolation', 'Unmonitored\nstations'],
        )
        ax.tick_params(axis='x', labelsize=7)
        ax.set_yticks([])
        ax.set_title(ANALYTE_LABELS[analyte])
        for j, val in enumerate(pivot):
            ax.text(j, 0, f'{val:.1f}%', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=axes, label='MAE reduction (%)')
    fig.suptitle('EcoHydroGraph versus snapshot: five-seed reduction')
    fig.savefig(OUT / 'reconstruction_improvement.pdf', dpi=220)
    plt.close(fig)

    # Fixed descriptive station: lexicographically first station with test cells
    # in every analyte for E2a. This is a display choice, not model selection.
    e2 = product[(product['mask'] == 'e2a_strict') & product['split'].eq('test')]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, analyte in zip(axes, ANALYTES):
        candidates = e2[e2.analyte.eq(analyte)].groupby('station').size()
        station = candidates.sort_values(ascending=False).index.sort_values()[0]
        d = product[(product['analyte'] == analyte) & (product['mask'] == 'e2a_strict') & (product['station'] == station)].copy()
        d['month'] = pd.to_datetime(d.month)
        observed = d[d['observed']]
        ax.plot(d.month, d.y_pred, color='#1f77b4', lw=1.2, label='EcoHydroGraph median')
        ax.fill_between(d.month, d.y_pred - d.seed_spread_sd, d.y_pred + d.seed_spread_sd,
                        color='#1f77b4', alpha=.15, label='seed ±1 SD')
        ax.scatter(observed.month, observed.y_true, s=6, color='#222222', alpha=.45, label='observed')
        ax.set_ylabel(ANALYTE_LABELS[analyte])
    axes[0].set_title('Unobserved-period extrapolation: station reconstructions')
    axes[-1].set_xlabel('Month')
    axes[0].legend(ncol=3, fontsize=8)
    fig.savefig(OUT / 'unobserved_period_traces.pdf', dpi=220)
    plt.close(fig)

    # E2a spatial error maps, one panel per analyte.
    d = station_error[station_error['mask'].eq('e2a_strict')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, analyte in zip(axes, ANALYTES):
        x = d[d.analyte.eq(analyte)]
        sc = ax.scatter(x.dec_long_va, x.dec_lat_va, c=x.mae, s=14, cmap='magma', alpha=.85)
        ax.set_title(ANALYTE_LABELS[analyte]); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        fig.colorbar(sc, ax=ax, label='Test MAE')
    fig.suptitle('Unobserved-period extrapolation: spatial error')
    fig.savefig(OUT / 'unobserved_period_spatial_error.pdf', dpi=220)
    plt.close(fig)


def window_figure():
    diag = pd.read_csv(ROOT / "t8_synthesis/diagnostic_summary.csv")
    windows = diag[diag.model_name.isin([
        "h2x_t", "h2x_t_lb3", "h2x_t_lb6", "h2x_t_current_only"
    ]) & diag["mask"].isin(["e2a_strict", "e2b_partial"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, analyte in zip(axes, ANALYTES):
        group = windows[windows.analyte.eq(analyte)]
        for mask, g in group.groupby("mask"):
            means = g.groupby("lookback").mae.mean()
            ax.plot(means.index, means.values, "o-", label=SCENARIOS[mask])
        ax.set_xticks([1, 3, 6, 12])
        ax.set_xlabel("History window (months)")
        ax.set_ylabel("MAE (native units)")
        ax.set_title(ANALYTE_LABELS[analyte])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(OUT / "history_window.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_figures(
        pd.read_parquet(ROOT / "t9_products/h2x_t_ensemble5_full_grid.parquet"),
        pd.read_csv(ROOT / "t9_products/test_station_error.csv", dtype={"station": str}),
    )
    window_figure()
