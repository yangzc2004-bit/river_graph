"""Compare legacy and refreshed support representations for frozen DOC experts.

No neural/forest model is fitted here. All planned within-expert basis contrasts
use fixed query cells and source-validation-selected support/mixing parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_daily_hydro_fallback_v1 import read_full_grid
from analyze_doc_encoder_residual_v1 import check_source_identity, compare
from analyze_doc_selective_residual_v1 import classification_metrics, recall_comparisons
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import read_bound_json, summarize_metrics

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_support_basis_v1")
ARMS = ("off", "current_only", "full_history")
SHAPES = ("constant", "legacy", "refreshed")
DIRECT_MODELS = tuple(f"{arm}_{shape}" for arm in ARMS for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS


def comparison_definitions():
    return [(f"{arm}{suffix}_refreshed_vs_legacy_k{k}", f"{arm}{suffix}_refreshed", k,
             f"{arm}{suffix}_legacy", k, "refreshed_support_basis" if not suffix else "refreshed_integrated_basis")
            for suffix in ("", "_integrated") for arm in ARMS for k in (1, 3, 5)]


def bound_file(run, name, completion, sources):
    path = run/name
    value = sha256(path)
    if completion["files"].get(name) != value:
        raise ValueError(f"Unbound or changed artifact: {path}")
    sources.append({"path": str(path), "sha256": value})
    return path


def verify_basis_archive(run, definition, completion, config, full, sources):
    """Check source readout lineage and label-free calendar normalization."""
    path = bound_file(run, "representations.npz", completion, sources)
    basis_run = Path(config["basis_run"])
    legacy_path = basis_run/"representations.npz"
    if sha256(legacy_path) != config["legacy_basis_hash"]:
        raise ValueError("Changed frozen legacy basis archive")
    sources.append({"path": str(legacy_path), "sha256": config["legacy_basis_hash"]})
    n_months = full.month.nunique()
    if len(full) % n_months:
        raise ValueError("Incomplete station calendar")
    n = len(full)//n_months
    anchors = np.linspace(0, n_months-1, min(n_months,32), dtype=np.int64)
    if (definition["anchor_count"] != 32 or definition["scale_floor"] != 1e-4
            or definition["readout_fitting"] != "none"
            or definition["parent_checkpoint_hashes"] != config["parent_checkpoint_hashes"]
            or np.asarray(definition["readout"]).shape != (64,2)
            or not np.isfinite(definition["readout"]).all()):
        raise ValueError("Changed fixed readout/normalization definition")
    with np.load(path, allow_pickle=False) as current, np.load(legacy_path, allow_pickle=False) as old:
        np.testing.assert_array_equal(current["constant"], old["constant"])
        np.testing.assert_array_equal(current["legacy"], old["gru_tuned_anchor"])
        if current["constant"].shape != (len(full),2) or current["legacy"].shape != (len(full),2):
            raise ValueError("Legacy or constant basis does not align with full grid")
        np.testing.assert_array_equal(current["anchor_months"], anchors)
        for arm in ARMS:
            raw, basis = current[f"{arm}_raw_basis"], current[f"{arm}_refreshed"]
            if (raw.shape != (n,n_months,2) or basis.shape != raw.shape
                    or not np.isfinite(raw).all() or not np.isfinite(basis).all()):
                raise ValueError("Invalid refreshed full-grid representation")
            anchor_raw = raw[:,anchors]
            mean = anchor_raw.mean(axis=1,keepdims=True)
            rms = np.sqrt(np.square(anchor_raw-mean).mean(axis=(1,2)))
            scale = np.maximum(rms,1e-4)
            # NumPy and torch float64 reductions can differ at round-off precision.
            np.testing.assert_allclose(current[f"{arm}_station_anchor_mean"],mean[:,0],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(current[f"{arm}_station_rms"],rms,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(current[f"{arm}_station_scale"],scale,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(current[f"{arm}_floor_hit"],rms<1e-4)
            np.testing.assert_allclose(basis,(raw-mean)/scale[:,None,None],rtol=1e-12,atol=1e-12)


def load_panel(root):
    frames, thresholds, sources, states, controls = [], {}, [], [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root/"runs"/f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run,sources)
            if ((config["split_seed"],config["seed"]) != (split,seed)
                    or config["experiment"] != "doc_daily_hydro_support_basis_v1"
                    or tuple(config["arms"]) != ARMS or tuple(config["basis_names"]) != SHAPES
                    or set(config["models"]) != set(MODELS) or tuple(config["k_values"]) != KS
                    or config["inference_roles"] != ["train"] or config["selection_role"] != "source_validation"
                    or config["target_analyte"] != "doc" or config["target_transform"] != "log1p"
                    or config["anchor_count"] != 32 or config["scale_floor"] != 1e-4
                    or config["backbone_retraining"] is not False or config["forest_retraining"] is not False
                    or config["readout_fitting"] is not False or config["gamma_k0"] != "frozen parent source-validation choice"):
                raise ValueError("Unexpected fixed-expert support-basis experiment settings")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold<0:
                raise ValueError("Invalid source Q90 threshold")
            thresholds[(split,seed)] = threshold
            for name in ("prior","source","basis"):
                path = Path(config[f"{name}_run"])/"complete.json"
                value = sha256(path)
                if value != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed frozen {name} package")
                sources.append({"path":str(path),"sha256":value})
            prior = Path(config["prior_run"])
            previous, old_config, old_completion = read_predictions(prior,sources)
            check_source_identity(config,old_config,("split_seed","seed","dataset_hash","mask_hash",
                "source_run","source_completion_hash","basis_run","basis_completion_hash","q90_threshold_train",
                "query_cells","extra_dim","daily_features_path","daily_features_hash","daily_metadata_path",
                "daily_metadata_hash","inference_roles"))
            if old_config["experiment"] != "doc_daily_hydro_memory_v1":
                raise ValueError("Parent is not the completed memory experiment")
            for arm in ARMS:
                path = prior/f"{arm}.pt"
                if sha256(path) != config["parent_checkpoint_hashes"][arm] or old_completion["files"][path.name] != sha256(path):
                    raise ValueError("Changed selected parent expert checkpoint")
                sources.append({"path":str(path),"sha256":config["parent_checkpoint_hashes"][arm]})
            basis_checkpoint = Path(config["basis_run"])/"memory.pt"
            if sha256(basis_checkpoint) != config["basis_checkpoint_hash"]:
                raise ValueError("Changed v4 fixed readout checkpoint")
            sources.append({"path":str(basis_checkpoint),"sha256":config["basis_checkpoint_hash"]})
            full = read_full_grid(run,config,completion,sources)
            if sha256(run/"full_grid.parquet") != config["parent_full_grid_hash"] or sha256(prior/"full_grid.parquet") != config["parent_full_grid_hash"]:
                raise ValueError("Frozen native full-grid predictions changed")
            sources.append({"path":str(prior/"full_grid.parquet"),"sha256":config["parent_full_grid_hash"]})
            definition = read_bound_json(run,"basis_definition.json",completion,sources)
            verify_basis_archive(run,definition,completion,config,full,sources)
            adapters = read_bound_json(run,"adapters.json",completion,sources)
            mixers = read_bound_json(run,"mixers.json",completion,sources)
            old_adapters = read_bound_json(prior,"adapters.json",old_completion,sources)
            old_mixers = read_bound_json(prior,"mixers.json",old_completion,sources)
            for arm in ARMS:
                for suffix, current_fits, old_fits in (("",adapters,old_adapters),("_integrated",mixers,old_mixers)):
                    for basis_name, old_basis in (("constant","constant"),("legacy","gru_tuned_anchor")):
                        name, old_name = f"{arm}{suffix}_{basis_name}",f"{arm}{suffix}_{old_basis}"
                        a = frame[frame.model_name.eq(name)].sort_values(["k","cell"])
                        b = previous[previous.model_name.eq(old_name)].sort_values(["k","cell"])
                        for field in ("cell","k","y_true","y_pred"):
                            np.testing.assert_array_equal(a[field],b[field])
                        if current_fits[name] != old_fits[old_name]:
                            raise ValueError("Unchanged-basis validation calibration failed parent replication")
                        controls.append({"split_seed":split,"seed":seed,"model_name":name,"parent_model":old_name,
                            "control_role":"constant_or_legacy_parent","n_query_rows":len(a),"bitwise_exact":True})
                    constant = frame[frame.model_name.eq(f"{arm}{suffix}_constant")&frame.k.eq(0)].sort_values("cell")
                    field = f"{arm}_integrated_k0_pred" if suffix else f"{arm}_pred"
                    np.testing.assert_array_equal(constant.y_pred,full.loc[constant.cell,field])
                    for basis_name in ("legacy","refreshed"):
                        name = f"{arm}{suffix}_{basis_name}"
                        candidate = frame[frame.model_name.eq(name)&frame.k.eq(0)].sort_values("cell")
                        np.testing.assert_array_equal(candidate.cell,constant.cell)
                        np.testing.assert_array_equal(candidate.y_pred,constant.y_pred)
                        if suffix and current_fits[name]["gamma_k0"] != old_fits[f"{arm}_integrated_constant"]["gamma_k0"]:
                            raise ValueError("Frozen parent K0 ecological mixture changed")
                        controls.append({"split_seed":split,"seed":seed,"model_name":name,
                            "parent_model":f"{arm}{suffix}_constant","control_role":"k0_invariance",
                            "n_query_rows":len(candidate),"bitwise_exact":True})
            checks = read_bound_json(run,"legacy_checks.json",completion,sources)
            if len(checks) != 48 or not all(v["bitwise_exact"] for v in checks):
                raise ValueError("Incomplete runner parent-replication checks")
            bound_file(run,"source_validation.csv",completion,sources)
            states.append({"split_seed":split,"seed":seed,"config":config,"adapters":adapters,"mixers":mixers})
            frames.append(frame)
        if len({thresholds[(split,seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across training seeds")
    panel = pd.concat(frames,ignore_index=True)
    validate_panel(panel,expected_models=MODELS)
    return panel,thresholds,sources,states,pd.DataFrame(controls)


def selection_records(states, panel):
    choices, mixing, scores, adapter_scores = [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != set(DIRECT_MODELS) or set(run["mixers"]) != set(MIXED_MODELS):
            raise ValueError("Unexpected direct or integrated support-model names")
        for model, state in run["adapters"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong support selection role or K")
            expected_ridge = ["infinity"] if model.endswith("_constant") else [.1, 1, 10, "infinity"]
            if state["alpha_values"] != [0, .25, .5, .75, 1] or state["ridge_strengths"] != expected_ridge:
                raise ValueError("Changed support-calibration grid")
            for k in KS:
                candidates = [v for v in state["selection_scores"] if v["k"] == k and v["valid"]]
                if not candidates:
                    raise ValueError("No finite validation candidate for support adaptation")
                best = min(candidates, key=lambda v: (v["mae"], v["alpha"],
                    -float("inf") if v["ridge_strength"] == "infinity" else -v["ridge_strength"]))
                selected = state["selection_by_k"][str(k)]
                if selected != {key: best[key] for key in ("alpha", "ridge_strength")}:
                    raise ValueError("Support choice differs from source-validation candidate trace")
                choices.append({**identity, "model_name": model, "k": k, **selected,
                                "validation_mae": best["mae"], "n_validation_query": best["n_query_cells"]})
            adapter_scores.extend({**identity, "model_name": model, **v} for v in state["selection_scores"])
        for model, state in run["mixers"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong integrated selection role or K")
            gamma_k0 = state["gamma_k0"]
            initial = [r for r in state["gamma_scores"] if r["k"] == 0]
            if gamma_k0 != min(initial, key=lambda r: (r["mae"], r["gamma"]))["gamma"]:
                raise ValueError("K0 gamma differs from its source-validation choice")
            for k in KS:
                selected = state["selection_by_k"][str(k)]
                candidates = [r for r in state["gamma_scores"] if r["k"] == k]
                eligible = [r for r in candidates if k != 0 or r["gamma"] == gamma_k0]
                chosen = min(eligible, key=lambda r: (r["mae"], r["gamma"] != 0,
                                                      r["gamma"] != gamma_k0, r["gamma"]))
                if selected != {**chosen, "locked": k == 0}:
                    raise ValueError("Integrated choice differs from recorded validation losses")
                if selected["gamma"] == 0:
                    direct = model.replace("_integrated_", "_")
                    a = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                    b = current[current.model_name.eq(direct) & current.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
                mixing.append({**identity, "model_name": model, "gamma_k0": gamma_k0, **selected})
                scores.extend({**identity, "model_name": model, **row} for row in candidates)
    return tuple(pd.DataFrame(rows) for rows in (choices, mixing, scores, adapter_scores))


def write_report(out, curves, profiles, classification, effects, choices, mixing, controls, draws):
    lines = ["# Aligning few-shot support representations with selected DOC experts", "",
        "This experiment keeps the off, current-only and full-history neural experts, their native",
        "base predictions, the context forest and ecological memory fixed. No new neural or forest",
        "fit is performed. It compares three support representations for each direct and integrated expert:", "",
        "- constant: the existing station-offset calibration diagnostic;",
        "- legacy: the previously used v4 two-dimensional GRU support basis;",
        "- refreshed: the selected current expert's hidden states projected through the unchanged",
        "  v4 two-dimensional readout, with the same 32-calendar-anchor centering and station scalar RMS normalization.", "",
        "The refreshed basis has no newly fitted PCA or readout. Support/mixing parameters are selected",
        "on the same source-validation episodes and grids. The direct contrasts measure representation",
        "alignment together with alpha/ridge reselection; integrated contrasts additionally include",
        "ecological gamma reselection. They are interpreted in that order. The full-timeline anchor",
        "normalization is label-free but retrospective, as in the legacy protocol.", "",
        "All 18 models and K={0,1,3,5} are retained. The three support representations have exact",
        "K0 invariance. Constant and legacy predictions reproduce their corresponding parent-memory",
        f"products exactly ({len(controls)} copied/control model-run checks). There is no target-based",
        "choice of expert, basis, K, route or calibration parameter.", "",
        f"Intervals use {draws:,} paired whole-station bootstrap draws. Repeated station identities",
        "are resampled jointly across partitions. Cell-pooled seed scores average within each",
        "partition, then partitions receive equal weight. Q90 uses source-training thresholds with ties",
        "included. Recall and false-Q90 rate are reported together; precision is descriptive.", "",
        "These are previously examined development partitions. The eighteen fixed basis contrasts",
        "are reported together; their pointwise intervals do not constitute an independent external confirmation.", "",
        "## Complete K curves", "",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        selected = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | "+" | ".join(f"{selected.loc[k, 'mae']:.6f}" for k in KS)
                     +f" | {selected.loc[5, 'rmse']:.6f} | {selected.loc[5, 'r2']:.6f} |")
    for role, title in (("refreshed_support_basis", "Direct support-basis contrasts"),
                        ("refreshed_integrated_basis", "Integrated support-basis contrasts")):
        lines += ["", f"## {title}", "",
            "| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |",
            "|---|---:|---:|---:|---:|---:|"]
        for row in effects[effects.comparison_role.eq(role)&effects.metric.eq("mae")&effects.region.eq("overall")].itertuples():
            group = effects[effects.comparison.eq(row.comparison)]

            def take(region, metric, group=group):
                return group[group.region.eq(region)&group.metric.eq(metric)].iloc[0]

            def fmt(value, scale=1):
                if value.status != "estimated":
                    return "not estimable"
                return f"{scale*value.delta_value:+.6f} [{scale*value.delta_ci_low:+.6f}, {scale*value.delta_ci_high:+.6f}]"

            lines.append(f"| {row.comparison} | {fmt(row)} | {fmt(take('q90', 'mae'))} | "
                         f"{fmt(take('nontail', 'mae'))} | {fmt(take('q90', 'q90_recall'),100)} | "
                         f"{fmt(take('nontail', 'q90_false_positive_rate'),100)} |")
    lines += ["", "## Tail error and classification at K5", "",
        "| Model | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        f=profiles[profiles.model_name.eq(model)&profiles.k.eq(5)].set_index("region")
        c=classification[classification.model_name.eq(model)&classification.k.eq(5)].iloc[0]
        lines.append(f"| {model} | {f.loc['q90','mae']:.6f} | {f.loc['q90','signed_bias']:+.6f} | "
                     f"{f.loc['nontail','mae']:.6f} | {f.loc['nontail','signed_bias']:+.6f} | "
                     f"{100*c.q90_recall:.3f}% | {100*c.q90_precision:.3f}% | {100*c.q90_false_positive_rate:.3f}% |")
    lines += ["", "## Source-validation calibration choices", "",
        "The exact per-run alpha/ridge/gamma choices and every gamma candidate score are preserved",
        "in the CSVs. Prediction changes at positive K can include parameter reselection as well as",
        "the changed basis. K0 equality verifies that this experiment does not change native reconstruction.", "",
        "| Model | K | Alpha counts | Ridge counts |", "|---|---:|---|---|"]
    for (model,k),g in choices.groupby(["model_name","k"],sort=False):
        alpha=", ".join(f"{v}: {n}" for v,n in g.alpha.value_counts().sort_index().items())
        ridge=", ".join(f"{v}: {n}" for v,n in g.ridge_strength.value_counts().items())
        lines.append(f"| {model} | {k} | {alpha} | {ridge} |")
    lines += ["", "| Integrated model | K | Gamma counts |", "|---|---:|---|"]
    for (model,k),g in mixing.groupby(["model_name","k"],sort=False):
        counts=", ".join(f"{v}: {n}" for v,n in g.gamma.value_counts().sort_index().items())
        lines.append(f"| {model} | {k} | {counts} |")
    lines += ["", "Partition/seed directions and station gain/harm concentration are included in the companion tables.",
        "A small or sign-changing estimate is not an equivalence test. No new ranking or calibration rule is fitted by this analysis.", ""]
    (out/"findings.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--bootstrap-draws",type=int,default=5000)
    args=parser.parse_args()
    if args.bootstrap_draws<1:
        raise ValueError("Bootstrap draws must be positive")
    panel,thresholds,sources,states,controls=load_panel(args.root)
    choices,mixing,gamma_scores,adapter_scores=selection_records(states,panel)
    runs,splits,curves=summarize_metrics(panel,thresholds)
    profile_runs,profile_splits,profiles=error_profiles(panel,thresholds)
    class_runs,class_splits,classification=classification_metrics(panel,thresholds)
    definitions=comparison_definitions()
    effects,directions,partitions,stations,global_stations,concentration=compare(
        panel,thresholds,args.bootstrap_draws,definitions=definitions)
    effects=pd.concat([effects,recall_comparisons(panel,thresholds,args.bootstrap_draws,definitions)],ignore_index=True)
    out=args.root/"analysis";out.mkdir(parents=True,exist_ok=True)
    outputs=(("metrics_by_run",runs),("metrics_by_partition",splits),("k_curves",curves),
        ("error_profiles_by_run",profile_runs),("error_profiles_by_partition",profile_splits),("error_profiles",profiles),
        ("classification_by_run",class_runs),("classification_by_partition",class_splits),("classification",classification),
        ("comparisons",effects),("directions_by_seed",directions),("directions_by_partition",partitions),
        ("station_responses",stations),("global_station_contributions",global_stations),("gain_loss_concentration",concentration),
        ("adapter_choices",choices),("adapter_scores",adapter_scores),("mixing_choices",mixing),("gamma_scores",gamma_scores),("parent_replication",controls))
    for name,frame in outputs:
        frame.to_csv(out/f"{name}.csv",index=False)
    write_report(out,curves,profiles,classification,effects,choices,mixing,controls,args.bootstrap_draws)
    hashes={f"{name}.csv":sha256(out/f"{name}.csv") for name,_ in outputs}
    hashes["findings.md"]=sha256(out/"findings.md")
    for name in ("analyze_doc_daily_hydro_fallback_v1.py","analyze_doc_encoder_residual_v1.py",
                 "analyze_doc_selective_residual_v1.py","analyze_doc_tail_residual_v1.py",
                 "analyze_unified_doc_spatial.py","analyze_unified_doc_spatial_v2.py","analyze_unified_doc_spatial_v3.py"):
        path=Path(__file__).parent/name;sources.append({"path":str(path),"sha256":sha256(path)})
    (out/"analysis_manifest.json").write_text(json.dumps({
        "analysis_script":str(Path(__file__)),"analysis_script_sha256":sha256(Path(__file__)),
        "bootstrap_draws":args.bootstrap_draws,"models":MODELS,"k_values":KS,"comparison_count":len(definitions),
        "sources":sources,"outputs":hashes,"role":"same-cohort support-representation development; no target model selection",
        "estimand":"cell-pooled within seed; seed mean within partition; equal partition mean"},indent=2)+"\n")
    print(curves[curves.k.isin((0,5))][["model_name","k","mae","rmse"]].to_string(index=False))
    print(f"Saved all 18 support models and {len(definitions)} fixed contrasts to {out}")


if __name__=="__main__":
    main()
