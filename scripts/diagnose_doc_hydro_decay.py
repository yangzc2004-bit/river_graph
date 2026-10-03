"""Inspect saved DOC-age decay using label-free visibility and river inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.graph_upgrade_v2 import observation_statistics
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    fold_split,
    role_visible,
    station_folds,
)

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
ARMS = ("off", "current_only", "full_history")
LAGS = (1, 3, 6, 11)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    flat = np.asarray(values).ravel()
    return {"n_values": int(flat.size), "mean": float(flat.mean()),
            "minimum": float(flat.min()), "p10": float(np.quantile(flat, .1)),
            "median": float(np.median(flat)), "p90": float(np.quantile(flat, .9)),
            "maximum": float(flat.max()), "fraction_below_half": float((flat < .5).mean()),
            "fraction_one": float((flat == 1).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sources = {}

    def bind(path, expected=None, scope=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"changed input: {path}")
        sources[str(path)] = {"sha256": actual, "scope": scope}
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    rows, channels, coefficients = [], [], []
    for partition in (142, 143, 144):
        for seed in (42, 43, 44):
            run = args.root / "runs" / f"split{partition}_seed{seed}"
            complete = read(run / "complete.json")
            config = read(run / "config.json", complete["files"]["config.json"])
            dataset = torch.load(bind(config["dataset_path"], config["dataset_hash"],
                                      "Dimensions, station/month identities and edge_index only; no target values"),
                                 weights_only=False, map_location="cpu")
            n, months = len(dataset["site_no"]), len(dataset["months"])
            edges = torch.as_tensor(dataset["edge_index"], dtype=torch.long)
            del dataset
            with np.load(bind(config["mask_path"], config["mask_hash"]), allow_pickle=False) as archive:
                split = {name: archive[name] for name in archive.files}
            oof = Path(config["oof_run"])
            oc = read(oof / "complete.json", config["oof_completion_hash"])
            records = read(oof / "oof_folds.json", oc["files"]["oof_folds.json"])
            folds = [np.asarray(record["held_station_ids"]) for record in records]
            expected = station_folds(np.asarray(split["train"]), months, seed)
            if not all(np.array_equal(a, b) for a, b in zip(folds, expected, strict=True)):
                raise ValueError("recorded source folds disagree with expert station-fold definition")
            dummy = torch.zeros((n, months), dtype=torch.float32)

            def inputs(view, n=n, months=months, dummy=dummy, edges=edges):
                visible = torch.as_tensor(role_visible(view, FIT_ROLES, (n, months)))
                stats = observation_statistics(dummy, visible, edges)
                # Exact raw_temporal_features support: local visibility, upstream, downstream-zero.
                return torch.stack([stats[1], stats[3], stats[-2], torch.zeros_like(stats[-1])], -1).permute(1, 0, 2).numpy()

            full = inputs(split)
            source_ids = np.unique(np.asarray(split["train"]) // months)
            source = np.empty((len(source_ids), months, 4), dtype=np.float32)
            for held in folds:
                source[np.searchsorted(source_ids, held)] = inputs(fold_split(split, held, months))[held]
            views = {"source_fold_hidden": (source_ids, source, np.asarray(split["train"]))}
            for role in ("val", "test"):
                station_ids = np.unique(np.asarray(split[role]) // months)
                _, query = support_query_cells(split, target_role=role, k=0, n_months=months)
                views["validation" if role == "val" else "target"] = (station_ids, full[station_ids], query)
            if any(np.any(x[..., 1]) for _, x, _ in views.values()):
                raise ValueError("the inspected station views must have no visible local DOC")
            for arm in ARMS:
                payload = torch.load(bind(run / f"{arm}.pt", complete["files"][f"{arm}.pt"],
                                          "Selected model coefficients only"), weights_only=True, map_location="cpu")
                weight = payload["decay"]["weight"].numpy()
                bias = payload["decay"]["bias"].numpy()
                projection = payload.get("hydro_projection", {}).get("weight")
                projection_norm = (np.linalg.norm(projection.numpy(), axis=1) if projection is not None
                                   else np.zeros(weight.shape[0]))
                for channel in range(weight.shape[0]):
                    coefficients.append({"split_seed": partition, "seed": seed, "arm": arm,
                                         "channel": channel, "age_weight": weight[channel, 0],
                                         "local_weight": weight[channel, 1], "upstream_weight": weight[channel, 2],
                                         "downstream_weight": weight[channel, 3], "bias": bias[channel],
                                         "hydro_projection_row_norm": projection_norm[channel]})
                for role, (station_ids, x, cells) in views.items():
                    cell_stations = np.searchsorted(station_ids, cells // months)
                    cell_months = cells % months
                    query_mask = np.zeros(x.shape[:2], dtype=bool)
                    query_mask[cell_stations, cell_months] = True
                    counterfactual = x.copy()
                    counterfactual[..., 0] = 0
                    for scenario, features in (("actual_doc_age", x), ("age_zero", counterfactual)):
                        # Torch float32 matches the saved recurrent calculation.
                        logits = torch.nn.functional.linear(torch.from_numpy(features),
                                                            torch.from_numpy(weight), torch.from_numpy(bias))
                        gamma = torch.exp(-torch.relu(logits)).numpy()
                        common = {"split_seed": partition, "seed": seed, "arm": arm, "role": role,
                                  "scenario": scenario, "n_stations": len(station_ids)}
                        for scope, chosen in (("calendar_grid", np.ones(x.shape[:2], bool)),
                                              ("observed_cells" if role == "source_fold_hidden" else "fixed_query_cells", query_mask)):
                            base = {**common, "scope": scope, "lag": 1}
                            rows.append({**base, **distribution(gamma[chosen])})
                            if scope == "calendar_grid":
                                for channel in range(gamma.shape[-1]):
                                    channels.append({**base, "channel": channel,
                                                     "hydro_projection_row_norm": projection_norm[channel],
                                                     **distribution(gamma[..., channel])})
                        for lag in LAGS[1:]:
                            # Innovation at t-lag encounters gamma[t-lag+1], ..., gamma[t].
                            retention = np.ones((len(station_ids), months-lag, gamma.shape[-1]), np.float32)
                            for offset in range(lag):
                                retention *= gamma[:, lag-offset:months-offset]
                            for scope, selected in (("calendar_grid", np.ones(retention.shape[:2], bool)),
                                                    ("observed_cells" if role == "source_fold_hidden" else "fixed_query_cells", query_mask[:, lag:])):
                                rows.append({**common, "scope": scope, "lag": lag,
                                             **distribution(retention[selected])})
                        del gamma
            print(f"Inspected label-free inputs and selected weights: {run.name}", flush=True)

    output = args.root / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    frame, channel_frame, coeff = pd.DataFrame(rows), pd.DataFrame(channels), pd.DataFrame(coefficients)
    frame.to_csv(output / "hydro_decay_retention.csv", index=False)
    channel_frame.to_csv(output / "hydro_decay_channels.csv", index=False)
    coeff.to_csv(output / "hydro_decay_coefficients.csv", index=False)
    average = frame.groupby(["arm", "role", "scope", "scenario", "lag"])[["mean", "p10", "median", "p90", "fraction_below_half", "fraction_one"]].mean().reset_index()
    average.to_csv(output / "hydro_decay_summary.csv", index=False)
    lines = ["# DOC-age attenuation of the shared hydrological memory", "",
             "This diagnostic reads selected checkpoint weights, river edges and visibility masks. All DOC values are replaced with zero before the observation-statistics function is called. No prediction products or target labels are read. Source stations use their recorded station-fold-hidden input views; validation and target station inputs use the fixed train/context visibility. The latter two groups are input grids, not outcome evidence.", "",
             "## Exact recurrence", "",
             r"For valid month $s$, $x_s=e_s+P d_s$ and $h_s=\mathrm{GRU}([x_s,1],\gamma_s\odot h_{s-1})$, where $\gamma_s=\exp[-\mathrm{ReLU}(W[a_s^{DOC},v_s^{local},v_s^{up},0]+b)]$. Current-only adds $Pd_s$ at the final step; full-history adds it throughout the 12-month causal window. Daily information enters unattenuated at its own step, but any part stored in the shared state subsequently encounters DOC-age decay.", "",
             r"At never-visible stations, $a_s^{DOC}=\log(1+s)/\log(13)$ for one-based calendar index $s$. It advances from the beginning of the dataset rather than measuring elapsed time since a discharge observation. Downstream support is zeroed in the actual KGML input path. The decay layer has 64 latent channels; these are not attention heads.", "",
             "## Measured explicit retention", "",
             "Values below average the nine saved models equally. Quantiles are calculated over station-month-channel values within each package, then averaged across packages; package and individual-channel distributions remain in the CSV files.", "",
             "| Arm | Input group | Scope | Mean gamma | p10 / median / p90 | Mean 3-step | Mean 6-step | Mean 11-step | Age-zero mean gamma |",
             "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for role in ("source_fold_hidden", "validation", "target"):
            for scope in ("calendar_grid", "observed_cells" if role == "source_fold_hidden" else "fixed_query_cells"):
                group = average[(average.arm == arm) & (average.role == role) & (average.scope == scope)]
                actual = group[group.scenario == "actual_doc_age"].set_index("lag")
                zero = group[(group.scenario == "age_zero") & (group.lag == 1)].iloc[0]
                first = actual.loc[1]
                lines.append(f"| {arm} | {role} | {scope} | {first['mean']:.4f} | {first.p10:.4f} / {first['median']:.4f} / {first.p90:.4f} | {actual.loc[3,'mean']:.4f} | {actual.loc[6,'mean']:.4f} | {actual.loc[11,'mean']:.4f} | {zero['mean']:.4f} |")
    lines += ["", "The one-step distribution includes all valid calendar months. Multi-step products exclude the beginning-of-record months without the requested history; fixed-query summaries use only their eligible query months. A lag-L quantity multiplies the L explicit decay factors after an input at t−L. It is not an effective prediction coefficient or the full GRU Jacobian.", "",
              "## Learned channels", ""]
    for arm in ARMS:
        c = coeff[coeff.arm == arm]
        lines.append(f"- {arm}: positive DOC-age coefficients in {int((c.age_weight > 0).sum())}/{len(c)} saved channels; range {c.age_weight.min():.5f} to {c.age_weight.max():.5f}. Hydro-projection row norms range {c.hydro_projection_row_norm.min():.5f}–{c.hydro_projection_row_norm.max():.5f}.")
    lines += ["", "## Interpretation and focused next comparison", "",
              "The learned operators retain a real DOC-age-dependent attenuation. This is a semantic coupling between hydrological persistence and the target-observation clock, not a source-to-target clock mismatch: source-fold loss stations, held validation stations and target stations all have their entire DOC series hidden in these recurrent inputs. These multiplicative factors alone cannot establish that prediction accuracy is limited by decay: GRU update/reset gates, recurrent matrices and the head can compensate. Setting age to zero here changes only a diagnostic input; no predictions, model selection or performance claims are made from it.", "",
              "The most focused next integration check is to refresh the station support-adaptation basis from the updated hydrological hidden representation. Current K0 predictions use the newly trained state, whereas K>0 shape adaptation still uses the older frozen v4 GRU basis. Compare old and refreshed representations under matched two-dimensional source-only basis construction, normalization and source-validation alpha/ridge selection, retaining both the off and full-history experts. This addresses an explicit representation disconnect without fitting another recurrent model. A refreshed basis must not be picked using target outcomes.", "",
              "If a subsequent storage experiment is needed, retain the original DOC/ecology recurrent path and isolate daily hydrology in a separate storage path, with matched current-only and historical controls using the same added parameters. A whole-state age-zero replacement changes all ecological and DOC memory and deactivates 64 age coefficients, so equal allocated parameter count would not mean equal effective capacity. A new uncapped hydrological-age clock also adds a summary beyond the 12-month window and must be identified as extra information. The present diagnostic supports testing the coupling, not assuming that bypassing it will improve predictions."]
    (output / "hydro_decay.md").write_text("\n".join(lines) + "\n")
    for path in (Path(__file__), Path("src/river_graph/experiments/graph_upgrade_v2.py"),
                 Path("src/river_graph/models/raw_temporal_features.py"),
                 Path("src/river_graph/models/kgml_local_transport.py"),
                 Path("src/river_graph/models/encoder_native_residual.py")):
        bind(path, scope="Diagnostic generator or recurrence/input definition")
    (output / "hydro_decay_sources.json").write_text(json.dumps({"read_scope": "label-free only", "sources": sources}, indent=2) + "\n")
    print(f"Wrote bounded decay diagnostic to {output}")


if __name__ == "__main__":
    main()
