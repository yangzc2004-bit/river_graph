"""Evaluate the frozen Stage-2 information decomposition from metric tables.

This command is deliberately metrics-only.  Stage-2B and Stage-2C have
already opened hidden query labels in their own final evaluators; this script
does not load datasets, predictions, or labels.  It only pairs the two
evaluators' task-level MAE tables under the frozen task key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.transfer import file_hash

BASELINE_DIR = Path("experiments/phase4_transfer/stage2b_cross_basin_v1")
CONTROLS_DIR = Path("experiments/phase4_transfer/stage2c_controls_v1_1")
SPEC = Path("experiments/phase4_transfer/stage2_information_decomposition_v1_spec.md")
OUT_DIR = Path("experiments/phase4_transfer/stage2_information_decomposition_v1")
TASK_KEYS = ["analyte", "basin", "task_seed", "task_index", "month", "k"]
BASELINE_METHODS = (
    "climatology",
    "eco_month_climatology",
    "local_mean",
    "mean_bias",
    "analytic_blend",
)
CONTROL_ARMS = ("ecorf", "h2x_full", "h2x_no_graph", "h2_no_ecology")
FAMILY = "same_month_spatial_huc6_leaveout"


def _canonical_basin(value: object) -> str:
    text = str(value)
    return text.zfill(6) if text.isdigit() else text


def _read_metrics(path: Path, source: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = set(TASK_KEYS) | {"model_name", "mae"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} metrics missing columns: {missing}")
    forbidden = {"y_true", "target_query_y", "query_labels"}
    leaked = sorted(forbidden.intersection(frame.columns))
    if leaked:
        raise ValueError(f"{source} metrics contain forbidden label columns: {leaked}")
    out = frame.copy()
    out["analyte"] = out["analyte"].astype(str)
    out["basin"] = out["basin"].map(_canonical_basin)
    out["task_seed"] = out["task_seed"].astype(int)
    out["task_index"] = out["task_index"].astype(int)
    out["k"] = out["k"].astype(int)
    out["month"] = out["month"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["mae"] = pd.to_numeric(out["mae"], errors="coerce")
    if out.empty or not np.isfinite(out["mae"]).all() or (out["mae"] < 0).any():
        raise ValueError(f"{source} metrics contain no rows or invalid MAE")
    if out[TASK_KEYS + ["model_name"]].duplicated().any():
        raise ValueError(f"{source} metrics contain duplicate model/task rows")
    return out


def _model(frame: pd.DataFrame, name: str, *, k: int | None = None) -> pd.DataFrame:
    out = frame[frame["model_name"].eq(name)].copy()
    if k is not None:
        out = out[out["k"].eq(k)]
    if out.empty:
        raise ValueError(f"missing model {name!r} (k={k})")
    return out


def _pair(
    comparator: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    comparator_name: str,
    reference_name: str,
    label: str,
    k: int | None,
) -> pd.DataFrame:
    left = comparator[TASK_KEYS + ["mae"]].rename(columns={"mae": "comparator_mae"})
    right = reference[TASK_KEYS + ["mae"]].rename(columns={"mae": "reference_mae"})
    merged = left.merge(right, on=TASK_KEYS, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[~merged["_merge"].eq("both"), TASK_KEYS].head(2).to_dict("records")
        raise ValueError(f"unpaired rows for {label}: {bad}")
    merged = merged.drop(columns="_merge")
    merged["delta_mae"] = merged["comparator_mae"] - merged["reference_mae"]
    merged["comparison"] = label
    merged["comparator"] = comparator_name
    merged["reference"] = reference_name
    merged["comparison_k"] = k if k is not None else "paired"
    return merged


def _bootstrap_months(frame: pd.DataFrame, value: str, *, reps: int, seed: int) -> tuple[float, float, float]:
    by_basin = {
        str(basin): group.groupby("month")[value].mean()
        for basin, group in frame.groupby("basin", sort=True)
    }
    if not by_basin:
        return float("nan"), float("nan"), float("nan")
    months = sorted(set().union(*(set(series.index) for series in by_basin.values())))
    observed = float(np.mean([series.mean() for series in by_basin.values()]))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    while len(draws) < reps:
        sample = rng.choice(months, size=len(months), replace=True)
        means = []
        for series in by_basin.values():
            values = series.reindex(sample).dropna()
            if values.empty:
                break
            means.append(float(values.mean()))
        if len(means) == len(by_basin):
            draws.append(float(np.mean(means)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def _summary(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    comparator: str,
    reference: str,
    reps: int,
    seed_start: int,
    k: int | str,
) -> list[dict]:
    rows = []
    for scope, groups in (("analyte_basin", frame.groupby(["analyte", "basin"], sort=True)),
                          ("analyte_pooled", frame.groupby("analyte", sort=True))):
        for key, group in groups:
            if scope == "analyte_basin":
                analyte, basin = key
                pooled_group = group
            else:
                analyte, basin = key, "all"
                pooled_group = group
            estimate, lo, hi = _bootstrap_months(
                pooled_group, value, reps=reps, seed=seed_start + len(rows)
            )
            n_month = int(pooled_group.groupby("month").ngroups)
            row = {
                "missingness_family": FAMILY,
                "scope": scope,
                "analyte": str(analyte),
                "basin": str(basin),
                "comparison": label,
                "comparator": comparator,
                "reference": reference,
                "k": k,
                "estimate": estimate,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_task_month": len(pooled_group),
                "n_unique_month": n_month,
                "bootstrap_reps": reps,
                "bootstrap_seed": seed_start + len(rows),
                "direction_improves": bool(estimate < 0),
                "ci_excludes_zero": bool(hi < 0 or lo > 0),
                "relative_reduction_pct": None,
            }
            if value == "delta_mae":
                baseline = float(pooled_group["reference_mae"].groupby(pooled_group["month"]).mean().mean())
                if baseline > 0:
                    row["relative_reduction_pct"] = float(-estimate / baseline * 100.0)
            rows.append(row)
    return rows


def _reference(frame: pd.DataFrame, *, model: str, label: str, reps: int, seed_start: int) -> list[dict]:
    selected = _model(frame, model, k=0).copy()
    selected["value"] = selected["mae"]
    rows = _summary(
        selected,
        label=label,
        value="value",
        comparator=model,
        reference="absolute_mae",
        reps=reps,
        seed_start=seed_start,
        k=0,
    )
    for row in rows:
        row["direction_improves"] = None
        row["ci_excludes_zero"] = None
        row["relative_reduction_pct"] = None
    return rows


def _append_comparison(rows: list[dict], paired: pd.DataFrame, *, label: str, comparator: str,
                       reference: str, reps: int, seed: int, k: int | str) -> None:
    rows.extend(_summary(
        paired,
        label=label,
        value="delta_mae",
        comparator=comparator,
        reference=reference,
        reps=reps,
        seed_start=seed,
        k=k,
    ))


def _verify_shared_keys(baselines: pd.DataFrame, controls: pd.DataFrame) -> None:
    base_keys = set(map(tuple, baselines[TASK_KEYS].drop_duplicates().to_numpy()))
    control_keys = set(map(tuple, controls[TASK_KEYS].drop_duplicates().to_numpy()))
    if base_keys != control_keys:
        raise ValueError("Stage-2B and Stage-2C task key sets differ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--controls-dir", default=str(CONTROLS_DIR))
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    if args.reps < 100:
        raise SystemExit("--reps must be at least 100")
    baseline_root, controls_root, out = map(Path, (args.baseline_dir, args.controls_dir, args.out_dir))
    spec_path = Path(args.spec)
    out.mkdir(parents=True, exist_ok=True)
    baselines = _read_metrics(baseline_root / "task_metrics.csv", "Stage-2B")
    controls = _read_metrics(controls_root / "task_metrics.csv", "Stage-2C")
    _verify_shared_keys(baselines, controls)
    if not set(baselines["model_name"].unique()).issuperset(BASELINE_METHODS):
        raise ValueError("Stage-2B baseline ladder is incomplete")
    if not set(controls["model_name"].unique()).issuperset(CONTROL_ARMS):
        raise ValueError("Stage-2C control arm set is incomplete")
    rows: list[dict] = []
    rows.extend(_reference(baselines, model="climatology", label="temporal_reference_climatology",
                           reps=args.reps, seed_start=42))
    eco = _pair(
        _model(baselines, "eco_month_climatology", k=0),
        _model(baselines, "climatology", k=0),
        comparator_name="eco_month_climatology",
        reference_name="climatology",
        label="heuristic_ecological_grouping",
        k=0,
    )
    _append_comparison(rows, eco, label="heuristic_ecological_grouping",
                       comparator="eco_month_climatology", reference="climatology",
                       reps=args.reps, seed=100, k=0)
    ecorf = _pair(
        _model(controls, "ecorf", k=0), _model(baselines, "climatology", k=0),
        comparator_name="ecorf", reference_name="climatology", label="ecorf_vs_temporal_reference", k=0,
    )
    _append_comparison(rows, ecorf, label="ecorf_vs_temporal_reference", comparator="ecorf",
                       reference="climatology", reps=args.reps, seed=200, k=0)

    for method in ("local_mean", "mean_bias", "analytic_blend"):
        pair = _pair(_model(baselines, method, k=5), _model(baselines, method, k=0),
                     comparator_name=method, reference_name=method,
                     label=f"support_increment_{method}", k=5)
        _append_comparison(rows, pair, label=f"support_increment_{method}", comparator=method,
                           reference=method, reps=args.reps, seed=300 + len(rows), k=5)
    for arm in CONTROL_ARMS:
        pair = _pair(_model(controls, arm, k=5), _model(controls, arm, k=0),
                     comparator_name=arm, reference_name=arm,
                     label=f"support_increment_{arm}", k=5)
        _append_comparison(rows, pair, label=f"support_increment_{arm}", comparator=arm,
                           reference=arm, reps=args.reps, seed=400 + len(rows), k=5)

    for k in (0, 5):
        graph = _pair(_model(controls, "h2x_full", k=k), _model(controls, "h2x_no_graph", k=k),
                      comparator_name="h2x_full", reference_name="h2x_no_graph",
                      label="graph_increment_h2x_full_vs_no_graph", k=k)
        _append_comparison(rows, graph, label="graph_increment_h2x_full_vs_no_graph",
                           comparator="h2x_full", reference="h2x_no_graph", reps=args.reps,
                           seed=500 + k, k=k)
        ecology = _pair(_model(controls, "h2x_full", k=k), _model(controls, "h2_no_ecology", k=k),
                        comparator_name="h2x_full", reference_name="h2_no_ecology",
                        label="combined_ecology_encoder_increment", k=k)
        _append_comparison(rows, ecology, label="combined_ecology_encoder_increment",
                           comparator="h2x_full", reference="h2_no_ecology", reps=args.reps,
                           seed=600 + k, k=k)

    # Frozen EcoRF does not consume target support.  Keep this as an explicit
    # implementation negative control and fail closed if it is not invariant.
    ecorf_support = _pair(_model(controls, "ecorf", k=5), _model(controls, "ecorf", k=0),
                          comparator_name="ecorf", reference_name="ecorf",
                          label="negative_control_ecorf_support_invariance", k=5)
    if not np.allclose(ecorf_support["delta_mae"], 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("EcoRF K=5/K=0 is not invariant")
    _append_comparison(rows, ecorf_support, label="negative_control_ecorf_support_invariance",
                       comparator="ecorf", reference="ecorf", reps=args.reps, seed=700, k=5)

    decomposition = pd.DataFrame(rows)
    decomposition.to_csv(out / "information_decomposition_v2.csv", index=False)
    curve = pd.concat([
        baselines.assign(source="stage2b"), controls.assign(source="stage2c")
    ], ignore_index=True)
    curve.groupby(["source", "model_name", "analyte", "basin", "k"], as_index=False).agg(
        mae=("mae", "mean"), n_task_month=("mae", "size")
    ).to_csv(out / "k_curve.csv", index=False)
    manifest = {
        "version": "phase4_stage2_information_decomposition_v1",
        "spec_sha256": file_hash(spec_path),
        "baseline_metrics_sha256": file_hash(baseline_root / "task_metrics.csv"),
        "controls_metrics_sha256": file_hash(controls_root / "task_metrics.csv"),
        "baseline_manifest_sha256": file_hash(baseline_root / "manifest.json"),
        "controls_manifest_sha256": file_hash(controls_root / "manifest.json"),
        "missingness_family": FAMILY,
        "secondary_missingness_families": "unavailable",
        "query_labels_opened_by_this_evaluator": False,
        "n_rows": len(decomposition),
        "bootstrap_reps": int(args.reps),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verdict = {
        "version": "phase4_stage2_information_decomposition_v1",
        "status": "completed_metrics_only_diagnostic",
        "training_started": False,
        "query_labels_opened_by_this_evaluator": False,
        "input_metrics_already_scored": True,
        "missingness_family": FAMILY,
        "secondary_missingness_families": "unavailable",
        "comparisons": sorted(decomposition["comparison"].unique().tolist()),
        "stage2_unlocked": False,
        "stage3_unlocked": False,
        "scientific_gate_status": "not_evaluated_by_this_diagnostic",
        "note": "Information deltas are predictive diagnostics; ecology grouping is heuristic and H2X ecology comparison is encoder-confounded.",
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (out / "STATUS.md").write_text(
        "# Stage 2 information decomposition\n\n"
        "Metrics-only diagnostic under the frozen same-month spatial HUC6 task. "
        "No query labels were reopened by this evaluator. Secondary missingness "
        "families are unavailable in the current task manifest. This artifact "
        "does not unlock Stage 2 or Stage 3.\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
