"""Read-only manuscript audit and exploratory temporal-skill diagnostics.

No training, checkpoint selection, protocol or primary endpoint is changed.
Within-station demeaning uses held-out truth only for evaluation; it is not an
inference input or a deployable prediction correction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riverdocnet_paper_zh/review"


def metrics(frame):
    y = frame.y_true.to_numpy(dtype=float)
    p = frame.y_pred.to_numpy(dtype=float)
    err = p-y
    numeric=pd.DataFrame({"station":frame.station.to_numpy(),"truth":y,"pred":p})
    yc = y-numeric.groupby("station").truth.transform("mean").to_numpy(dtype=float)
    pc = p-numeric.groupby("station").pred.transform("mean").to_numpy(dtype=float)
    sst = np.sum((y-y.mean())**2)
    within = np.sum(yc**2)
    log_frame=frame[["station"]].copy()
    log_frame["truth"]=np.log1p(y)
    log_frame["pred"]=np.log1p(p)
    log_yc=log_frame.truth-log_frame.groupby("station").truth.transform("mean")
    log_pc=log_frame.pred-log_frame.groupby("station").pred.transform("mean")
    log_within=float(np.sum(log_yc**2))
    ordered = frame.sort_values(["station", "month"])
    delta = ordered.groupby("station")[["month", "y_true", "y_pred"]].diff()
    delta = delta[delta.month == 1]
    corr = lambda a,b: float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-6 and np.std(b)>1e-6 else None
    return {
        "mae": float(np.abs(err).mean()),
        "log1p_mae": float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1-np.sum(err**2)/sst),
        "bias": float(err.mean()),
        "query_cells": len(frame),
        "query_stations": int(frame.station.nunique()),
        "observed_between_station_variance_fraction": float(1-within/sst),
        "station_centered_r2": float(1-np.sum((pc-yc)**2)/within),
        "station_centered_correlation": corr(pc,yc),
        "log1p_station_centered_r2": float(1-np.sum((log_pc-log_yc)**2)/log_within),
        "log1p_station_centered_correlation": corr(log_pc,log_yc),
        "predicted_to_observed_within_station_sd_ratio": float(np.sqrt(np.sum(pc**2)/within)),
        "consecutive_month_pair_count": len(delta),
        "monthly_difference_correlation": corr(delta.y_pred,delta.y_true),
        "monthly_difference_mae": float(np.abs(delta.y_pred-delta.y_true).mean()),
        "zero_monthly_change_mae": float(np.abs(delta.y_true).mean()),
    }


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    root = ROOT / "experiments/phase4_transfer/doc_unified_transfer_v2"
    reference = pd.read_csv(root/"analysis/by_fit.csv")
    reference = reference[reference.domain == "source_hidden_station"].set_index("id")
    rows, discrepancies = [], []
    for identifier,row in reference.iterrows():
        frame = pd.read_parquet(root/"fits"/identifier/"source_predictions.parquet")
        assert not frame.cell.duplicated().any()
        result = metrics(frame)
        for metric in ("mae","log1p_mae","rmse","r2","bias"):
            difference=abs(result[metric]-float(row[metric]))
            discrepancies.append(difference)
            assert difference < 1e-6,(identifier,metric,difference)
        rows.append({"id":identifier,"arm":row.arm,"split":int(row.split),"seed":int(row.seed),**result})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT/"random_station_diagnostics_by_fit.csv",index=False)
    summary = raw.groupby("arm").mean(numeric_only=True).drop(columns=["split","seed"]).reset_index()
    summary.to_csv(OUT/"random_station_diagnostics_summary.csv",index=False)

    regional=[]
    regions=("1013","1019","0708","1030","1101")
    roots={"direction_heads":"doc_river_relation_v2",
           "fixed_fusion":"doc_river_relation_v2",
           "no_time":"doc_information_ablation_v1"}
    for arm,experiment in roots.items():
        for seed in (42,43,44):
            files=[ROOT/f"experiments/phase4_transfer/{experiment}/fits/space_{region}_{arm}_s{seed}/predictions.parquet" for region in regions]
            if not all(path.exists() for path in files):
                raise FileNotFoundError([str(p) for p in files if not p.exists()])
            frame=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
            assert not frame.cell.duplicated().any()
            # The regional archive stores flattened cells and station indices.
            # Confirm the established 654-month grid before decoding calendar.
            assert np.array_equal(frame.cell.to_numpy()//654,frame.station.to_numpy())
            frame["month"]=frame.cell.to_numpy()%654
            regional.append({"arm":arm,"seed":seed,**metrics(frame)})
    region_frame=pd.DataFrame(regional)
    region_frame.to_csv(OUT/"regional_diagnostics_by_seed.csv",index=False)
    region_summary=region_frame.groupby("arm").mean(numeric_only=True).drop(columns="seed").reset_index()
    region_summary.to_csv(OUT/"regional_diagnostics_summary.csv",index=False)
    temporal_rows=[]
    for gap in (3,6,12):
        for arm in ("direction_heads","fixed_fusion"):
            for seed in (42,43,44):
                p=ROOT/f"experiments/phase4_transfer/doc_river_relation_v2/fits/time_g{gap}_{arm}_s{seed}/predictions.parquet"
                frame=pd.read_parquet(p)
                assert np.array_equal(frame.cell.to_numpy()//654,frame.station.to_numpy())
                frame["month"]=frame.cell.to_numpy()%654
                temporal_rows.append({"arm":arm,"gap_months":gap,"seed":seed,**metrics(frame)})
    temporal_raw=pd.DataFrame(temporal_rows)
    temporal_raw.to_csv(OUT/"time_gap_diagnostics_by_fit.csv",index=False)
    temporal_summary=temporal_raw.groupby(["arm","gap_months"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    temporal_summary.to_csv(OUT/"time_gap_diagnostics_summary.csv",index=False)
    benchmark_root=ROOT/"experiments/phase4_transfer/doc_network_work_v2/regions"
    benchmark=pd.concat([pd.read_parquet(benchmark_root/f"huc4_{region}/predictions.parquet")
                         for region in regions],ignore_index=True)
    benchmark["month"]=benchmark.cell.to_numpy()%654
    benchmark_rows=[]
    for (arm,seed),frame in benchmark.groupby(["model_name","seed"]):
        assert not frame.cell.duplicated().any()
        benchmark_rows.append({"arm":arm,"seed":seed,**metrics(frame)})
    benchmark_raw=pd.DataFrame(benchmark_rows)
    benchmark_raw.to_csv(OUT/"benchmark_diagnostics_by_seed.csv",index=False)
    benchmark_summary=benchmark_raw.groupby("arm").mean(numeric_only=True).drop(columns="seed").reset_index()
    benchmark_summary.to_csv(OUT/"benchmark_diagnostics_summary.csv",index=False)
    audit={"source_prediction_tables_recomputed":len(rows),
           "max_primary_metric_discrepancy":max(discrepancies),
           "diagnostic_scope":"exploratory observed-query evaluation; no changes to frozen primary endpoints",
           "centered_r2_definition":"1 - sum((pred - station_mean_pred - truth + station_mean_truth)^2) / sum((truth - station_mean_truth)^2)",
           "demeaning_is_evaluation_only":True,
           "correlation_near_constant_tolerance":1e-6,
           "monthly_pair_definition":"observed query pairs at the same station exactly one calendar month apart",
           "summary_rule":"arithmetic mean of fit-level diagnostic metrics; no prediction ensemble",
           "regional_parent_caution":"fixed_fusion and no_time share the early parent; direction_heads is the final spatial recipe"}
    benchmark_expected=pd.read_csv(ROOT/"experiments/phase4_transfer/doc_network_work_v2/analysis/summary.csv").set_index("model_name")
    benchmark_error=max(abs(float(row[metric])-float(benchmark_expected.loc[row["arm"],metric]))
                        for row in benchmark_summary.to_dict("records")
                        for metric in ("mae","log1p_mae","rmse","r2"))
    assert benchmark_error<1e-6,benchmark_error
    audit["benchmark_primary_metric_max_discrepancy"]=benchmark_error
    (OUT/"prediction_diagnostic_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    columns=["arm","mae","r2","station_centered_r2","log1p_station_centered_r2","station_centered_correlation",
             "predicted_to_observed_within_station_sd_ratio","monthly_difference_correlation",
             "monthly_difference_mae","zero_monthly_change_mae","consecutive_month_pair_count"]
    print("Random-station diagnostics")
    print(summary[columns].round(4).to_string(index=False))
    print("Regional diagnostics")
    print(region_summary[columns].round(4).to_string(index=False))
    print("Same-query benchmark diagnostics")
    print(benchmark_summary[columns].round(4).to_string(index=False))
    print("Temporal-gap diagnostics")
    print(temporal_summary[["arm","gap_months",*columns[1:]]].round(4).to_string(index=False))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
