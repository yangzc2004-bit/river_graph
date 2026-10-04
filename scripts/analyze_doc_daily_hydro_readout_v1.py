"""Compare fixed and source-trained support readouts for the frozen DOC expert.

No prediction expert is fitted here. All planned within-expert readout contrasts
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

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_readout_v1")
SHAPES = ("constant", "legacy", "refreshed_fixed", "refreshed_learned")
DIRECT_MODELS = tuple(f"off_{shape}_direct" for shape in SHAPES)
MIXED_MODELS = tuple(f"off_{shape}_integrated" for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS


def comparison_definitions():
    return [(f"off_refreshed_learned_vs_{reference}_{stage}_k{k}",
             f"off_refreshed_learned_{stage}", k, f"off_{reference}_{stage}", k,
             f"learned_readout_{stage}")
            for stage in ("direct", "integrated") for reference in ("refreshed_fixed", "legacy") for k in (3,5)]


def bound_file(run, name, completion, sources):
    path = run/name
    value = sha256(path)
    if completion['files'].get(name) != value:
        raise ValueError(f'Unbound or changed artifact: {path}')
    sources.append({'path':str(path),'sha256':value})
    return path


def verify_basis_archive(run, definition, completion, config, full, sources):
    """Check unchanged controls and label-free anchor normalization."""
    prior = Path(config['prior_run'])
    archive = bound_file(run,'representations.npz',completion,sources)
    previous_path = prior/'representations.npz'
    if sha256(previous_path) != config['parent_representation_hash']:
        raise ValueError('Frozen parent representation archive changed')
    sources.append({'path':str(previous_path),'sha256':config['parent_representation_hash']})
    months = full.month.nunique()
    if len(full)%months:
        raise ValueError('Incomplete full-grid calendar')
    n = len(full)//months
    initial = np.asarray(definition['initial_readout'])
    selected = np.asarray(definition['selected_readout'])
    if (definition['anchor_count'] != 32 or definition['scale_floor'] != 1e-4
            or definition['readout_fitting'] != 'source support episodes'
            or definition['normalization_role'] != 'retrospective feature-only record normalization'
            or definition['parent_checkpoint_hash'] != config['parent_checkpoint_hash']
            or initial.shape != (64,2) or selected.shape != (64,2)
            or not np.isfinite(initial).all() or not np.isfinite(selected).all()):
        raise ValueError('Changed selected-readout basis definition')
    anchors = np.linspace(0,months-1,min(months,32),dtype=np.int64)
    with np.load(archive,allow_pickle=False) as current, np.load(previous_path,allow_pickle=False) as old:
        for name, old_name in (('constant','constant'),('legacy','legacy'),('refreshed_fixed','off_refreshed')):
            np.testing.assert_array_equal(current[name],old[old_name].reshape(-1,2))
        np.testing.assert_array_equal(current['anchor_months'],anchors)
        for name in SHAPES:
            if current[name].shape != (len(full),2) or not np.isfinite(current[name]).all():
                raise ValueError('Invalid or misaligned support representation')
        for name in ('refreshed_fixed','refreshed_learned'):
            raw = current[f'{name}_raw_basis']
            if raw.shape != (n,months,2) or not np.isfinite(raw).all():
                raise ValueError('Invalid pre-normalization projection')
            anchor_raw = raw[:,anchors]
            mean = anchor_raw.mean(1,keepdims=True)
            rms = np.sqrt(np.square(anchor_raw-mean).mean((1,2)))
            scale = np.maximum(rms,1e-4)
            # Torch and NumPy float64 reductions can differ at round-off precision.
            np.testing.assert_allclose(current[f'{name}_station_anchor_mean'],mean[:,0],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(current[f'{name}_station_rms'],rms,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(current[f'{name}_station_scale'],scale,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(current[f'{name}_floor_hit'],rms<1e-4)
            np.testing.assert_allclose(current[name].reshape(n,months,2),
                                       (raw-mean)/scale[:,None,None],rtol=1e-12,atol=1e-12)
    return months


def load_panel(root):
    frames, thresholds, sources, states, controls = [], {}, [], [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root/'runs'/f'split{split}_seed{seed}'
            frame, config, completion = read_predictions(run,sources)
            if ((config['split_seed'],config['seed']) != (split,seed)
                    or config['experiment'] != 'doc_daily_hydro_readout_v1'
                    or config['arms'] != ['off'] or tuple(config['basis_names']) != SHAPES
                    or set(config['models']) != set(MODELS) or tuple(config['k_values']) != KS
                    or config['inference_roles'] != ['train'] or config['selection_role'] != 'source_validation'
                    or config['target_analyte'] != 'doc' or config['target_transform'] != 'log1p'
                    or config['anchor_count'] != 32 or config['scale_floor'] != 1e-4
                    or config['backbone_retraining'] is not False or config['forest_retraining'] is not False
                    or config['readout_fitting'] is not True or config['gamma_k0'] != 'frozen parent source-validation choice'
                    or config['epochs'] != 30 or config['patience'] != 5 or config['batch_size'] != 32
                    or config['learning_rate'] != .001 or config['gradient_clip_norm'] != 1
                    or config['trainable_parameter_count'] != 128 or config['train_k'] != [3,5]
                    or config['train_ridge'] != [1,10] or config['train_alpha'] != 1
                    or config['projection_batch_size'] != 2048 or config['hidden_batch_size'] != 2048):
                raise ValueError('Unexpected frozen-expert learned-readout experiment settings')
            threshold = float(config['q90_threshold_train'])
            if not np.isfinite(threshold) or threshold<0:
                raise ValueError('Invalid source-training Q90 threshold')
            thresholds[(split,seed)] = threshold
            for name in ('prior','source','basis','expert','oof'):
                path = Path(config[f'{name}_run'])/'complete.json'
                value = sha256(path)
                if value != config[f'{name}_completion_hash']:
                    raise ValueError(f'Changed frozen {name} package')
                sources.append({'path':str(path),'sha256':value})
            prior = Path(config['prior_run'])
            previous, old_config, old_completion = read_predictions(prior,sources)
            check_source_identity(config,old_config,('split_seed','seed','dataset_hash','mask_hash',
                'source_run','source_completion_hash','basis_run','basis_completion_hash','q90_threshold_train',
                'query_cells','daily_features_path','daily_features_hash','daily_metadata_path',
                'daily_metadata_hash','inference_roles'))
            if old_config['experiment'] != 'doc_daily_hydro_support_basis_v1':
                raise ValueError('Parent is not the completed fixed-support-basis study')
            if (config['expert_run'] != old_config['prior_run']
                    or config['expert_completion_hash'] != old_config['prior_completion_hash']):
                raise ValueError('Selected expert does not match fixed-support parent')
            for path, field in ((Path(config['expert_run'])/'off.pt','parent_checkpoint_hash'),
                                (Path(config['basis_run'])/'memory.pt','basis_checkpoint_hash')):
                if sha256(path) != config[field]:
                    raise ValueError('Changed frozen recurrent expert or initial readout checkpoint')
                sources.append({'path':str(path),'sha256':config[field]})
            full = read_full_grid(run,config,completion,sources)
            if (sha256(run/'full_grid.parquet') != config['parent_full_grid_hash']
                    or sha256(prior/'full_grid.parquet') != config['parent_full_grid_hash']):
                raise ValueError('Frozen native full-grid predictions changed')
            sources.append({'path':str(prior/'full_grid.parquet'),'sha256':config['parent_full_grid_hash']})
            definition = read_bound_json(run,'basis_definition.json',completion,sources)
            months = verify_basis_archive(run,definition,completion,config,full,sources)
            adapters = read_bound_json(run,'adapters.json',completion,sources)
            mixers = read_bound_json(run,'mixers.json',completion,sources)
            readout = read_bound_json(run,'readout.json',completion,sources)
            inputs = read_bound_json(run,'input_definition.json',completion,sources)
            bound_file(run,'readout.pt',completion,sources)
            bound_file(run,'readout_trace.csv',completion,sources)
            source_path = bound_file(run,'source_training.npz',completion,sources)
            if (inputs['forest_only_oof'] is not True or inputs['frozen_neural_source_trained'] is not True
                    or inputs['source_hidden_shape'] != [len(inputs['source_station_ids']),months,64]
                    or inputs['full_hidden_shape'] != [len(full)//months,months,64]):
                raise ValueError('Changed frozen source/full hidden input definition')
            with np.load(source_path,allow_pickle=False) as source:
                np.testing.assert_array_equal(source['source_station_ids'],inputs['source_station_ids'])
                np.testing.assert_array_equal(source['validation_station_ids'],inputs['validation_station_ids'])
                if (source['source_native'].shape != source['source_mask'].shape
                        or source['source_native'].shape != (len(inputs['source_station_ids']),months)
                        or not np.isfinite(source['source_native'][source['source_mask']]).all()
                        or not np.isnan(source['source_native'][~source['source_mask']]).all()):
                    raise ValueError('Invalid observed-only source episode base')
            old_adapters = read_bound_json(prior,'adapters.json',old_completion,sources)
            old_mixers = read_bound_json(prior,'mixers.json',old_completion,sources)
            for stage, fits, old_fits in (('direct',adapters,old_adapters),('integrated',mixers,old_mixers)):
                for basis, parent_basis in (('constant','constant'),('legacy','legacy'),('refreshed_fixed','refreshed')):
                    name = f'off_{basis}_{stage}'
                    parent_name = f"off{'_integrated' if stage=='integrated' else ''}_{parent_basis}"
                    a = frame[frame.model_name.eq(name)].sort_values(['k','cell'])
                    b = previous[previous.model_name.eq(parent_name)].sort_values(['k','cell'])
                    for field in ('cell','k','y_true','y_pred'):
                        np.testing.assert_array_equal(a[field],b[field])
                    if fits[name] != old_fits[parent_name]:
                        raise ValueError('Unchanged-basis validation choices failed parent replication')
                    controls.append({'split_seed':split,'seed':seed,'model_name':name,'parent_model':parent_name,
                        'control_role':'unchanged_basis_parent','n_query_rows':len(a),'bitwise_exact':True})
                for k in (0,1):
                    reference_name = f'off_legacy_{stage}'
                    reference = frame[frame.model_name.eq(reference_name)&frame.k.eq(k)].sort_values('cell')
                    for basis in SHAPES:
                        name = f'off_{basis}_{stage}'
                        candidate = frame[frame.model_name.eq(name)&frame.k.eq(k)].sort_values('cell')
                        np.testing.assert_array_equal(candidate.cell,reference.cell)
                        np.testing.assert_array_equal(candidate.y_pred,reference.y_pred)
                        if k==0:
                            field = 'off_integrated_k0_pred' if stage=='integrated' else 'off_pred'
                            np.testing.assert_array_equal(candidate.y_pred,full.loc[candidate.cell,field])
                        if stage=='integrated' and fits[name]['gamma_k0'] != old_mixers['off_integrated_constant']['gamma_k0']:
                            raise ValueError('K0 ecological mixture was not preserved')
                        controls.append({'split_seed':split,'seed':seed,'model_name':name,'parent_model':reference_name,
                            'control_role':f'k{k}_invariance','n_query_rows':len(candidate),'bitwise_exact':True})
            checks = read_bound_json(run,'control_checks.json',completion,sources)
            if len(checks)!=24 or not all(v['bitwise_exact'] for v in checks):
                raise ValueError('Incomplete fixed-control runner checks')
            bound_file(run,'source_validation.csv',completion,sources)
            states.append({'split_seed':split,'seed':seed,'config':config,'adapters':adapters,'mixers':mixers,'readout':readout})
            frames.append(frame)
        if len({thresholds[(split,seed)] for seed in SEEDS}) != 1:
            raise ValueError('Source Q90 differs across training seeds')
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
            expected_ridge = ["infinity"] if "_constant_" in model else [.1, 1, 10, "infinity"]
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
                    direct = model.removesuffix("_integrated")+"_direct"
                    a = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                    b = current[current.model_name.eq(direct) & current.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
                mixing.append({**identity, "model_name": model, "gamma_k0": gamma_k0, **selected})
                scores.extend({**identity, "model_name": model, **row} for row in candidates)
    return tuple(pd.DataFrame(rows) for rows in (choices, mixing, scores, adapter_scores))


def write_report(out, curves, profiles, classification, effects, choices, mixing, controls, training, draws):
    lines = ["# Source-trained readout for the fixed DOC support representation", "",
        "The off daily-hydrology expert, native predictions and ecological memory remain fixed.",
        "Four support representations are reported: constant station offset, legacy episodic basis,",
        "refreshed hidden states with the fixed v4 readout, and those same hidden states with a",
        "source-trained readout. Only source training fits the readout; source validation selects",
        "its checkpoint and calibration coefficients. Target outcomes never choose a readout or route.", "",
        "The learned-versus-fixed contrast asks whether training the projection improves adaptation",
        "on the selected expert's unchanged hidden states. Learned-versus-legacy asks whether the",
        "complete updated representation improves the existing station-support method. Direct",
        "contrasts include alpha/ridge reselection; integrated contrasts also include positive-K",
        "ecological gamma reselection. Native predictions and the K0 ecological mixture stay fixed.", "",
        "K0 has no support update. K1 has no shape correction: one centered support point has rank",
        "zero and the adapter explicitly enables the shape term only for K>1. These structural",
        "negative controls are reported without interpreting exact zeros as low statistical power.",
        f"The loader verifies {len(controls)} copied-prediction or invariance checks.", "",
        f"All eight planned contrasts use {draws:,} paired whole-station bootstrap draws. Repeated",
        "station identities are jointly resampled across partitions; seed means average within",
        "partitions, then partitions receive equal weight. Q90 is the source-training threshold",
        "including ties. Pointwise intervals are reported on reused development partitions.", "",
        "## Complete K curves", "",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        selected = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | "+" | ".join(f"{selected.loc[k,'mae']:.6f}" for k in KS)
                     +f" | {selected.loc[5,'rmse']:.6f} | {selected.loc[5,'r2']:.6f} |")
    for role,title in (("learned_readout_direct","Direct readout contrasts"),
                       ("learned_readout_integrated","Integrated readout contrasts")):
        lines += ["",f"## {title}","",
            "| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |",
            "|---|---:|---:|---:|---:|---:|"]
        for row in effects[effects.comparison_role.eq(role)&effects.metric.eq("mae")&effects.region.eq("overall")].itertuples():
            group=effects[effects.comparison.eq(row.comparison)]

            def take(region,metric,group=group):
                return group[group.region.eq(region)&group.metric.eq(metric)].iloc[0]

            def fmt(value,scale=1):
                if value.status!="estimated":
                    return "not estimable"
                return f"{scale*value.delta_value:+.6f} [{scale*value.delta_ci_low:+.6f}, {scale*value.delta_ci_high:+.6f}]"

            lines.append(f"| {row.comparison} | {fmt(row)} | {fmt(take('q90','mae'))} | "
                f"{fmt(take('nontail','mae'))} | {fmt(take('q90','q90_recall'),100)} | "
                f"{fmt(take('nontail','q90_false_positive_rate'),100)} |")
    lines += ["","## Tail and ordinary errors, with detection tradeoffs","",
        "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for k in (3,5):
            f=profiles[profiles.model_name.eq(model)&profiles.k.eq(k)].set_index("region")
            c=classification[classification.model_name.eq(model)&classification.k.eq(k)].iloc[0]
            lines.append(f"| {model} | {k} | {f.loc['q90','mae']:.6f} | {f.loc['q90','signed_bias']:+.6f} | "
                f"{f.loc['nontail','mae']:.6f} | {f.loc['nontail','signed_bias']:+.6f} | "
                f"{100*c.q90_recall:.3f}% | {100*c.q90_precision:.3f}% | {100*c.q90_false_positive_rate:.3f}% |")
    lines += ["","## Source-validation readout checkpoint selection","",
        "The saved checkpoint choices below describe source-validation optimization, not held-out",
        "target performance. Epoch zero is retained as a fixed-readout fallback.",
        "The 128-parameter projection is trained with equal-station native MAE averaged over",
        "K={3,5} and ridge={1,10}, at alpha=1; checkpoint validation pools fixed query cells",
        "over the same four conditions. Final calibration then uses the unchanged alpha/ridge grid.",
        "The source baseline combines forest OOF predictions with a source-trained neural correction;",
        "it is not a fully OOF neural expert. Source episode loss is a supervised training diagnostic.","",
        "| Partition | Seed | Selected epoch | Epochs run | Initial validation loss | Selected validation loss |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in training.itertuples():
        lines.append(f"| {row.split_seed} | {row.seed} | {row.best_epoch} | {row.epochs_run} | "
                     f"{row.initial_validation_loss:.6f} | {row.selected_validation_loss:.6f} |")
    lines += ["","Selected support alpha/ridge values, all direct candidate scores, ecological gamma",
        "choices and all gamma scores are preserved in the CSVs. Full K curves, partition/seed",
        "directions, transformed errors and station gain/harm concentration are also retained.",
        "No target-based K switch or model selection is produced by this analyzer.",""]
    (out/"findings.md").write_text("\n".join(lines))


def training_records(states):
    rows = []
    for run in states:
        saved = run["readout"]
        config, protocol, trace = saved["config"], saved["protocol"], saved["trace"]
        if (saved["model_class"] != "EpisodicSupportReadout" or saved["hidden_size"] != 64
                or saved["output_dim"] != 2 or saved["trainable_parameter_count"] != 128
                or config["anchor_count"] != 32 or config["scale_floor"] != 1e-4
                or config["seed"] != run["seed"] or config["max_epochs"] != 30
                or config["patience"] != 5 or config["lr"] != 1e-3 or config["batch_size"] != 32
                or protocol["selection_role"] != "source_validation" or protocol["alpha"] != 1
                or protocol["k_values"] != [3,5] or protocol["ridge_strengths"] != [1.,10.]
                or protocol["training_loss"] != "raw_mae_equal_station_mean_k_ridge"
                or protocol["validation_loss"] != "raw_mae_pooled_query_mean_k_ridge"
                or protocol["hidden_experts_updated"] is not False or protocol["pca_whitening_qr"] is not False):
            raise ValueError("Saved episodic readout differs from the fixed source-training protocol")
        if [v["epoch"] for v in trace] != list(range(saved["epochs_run"]+1)):
            raise ValueError("Incomplete readout checkpoint trace")
        chosen = min(trace,key=lambda v:(v["validation_mae"],v["epoch"]))
        if (chosen["epoch"] != saved["best_epoch"] or chosen != saved["validation_metrics"]
                or not np.isfinite([v["validation_mae"] for v in trace]).all()):
            raise ValueError("Readout checkpoint differs from its source-validation trace")
        for key in ("readout_parameter_distance","readout_norm","delta_norm"):
            if not np.isfinite(saved[key]) or saved[key]<0:
                raise ValueError("Invalid readout coefficient diagnostic")
        if saved["best_epoch"] == 0 and saved["readout_parameter_distance"] != 0:
            raise ValueError("Epoch-zero fallback altered the fixed readout")
        rows.append({"split_seed":run["split_seed"],"seed":run["seed"],
            "best_epoch":saved["best_epoch"],"epochs_run":saved["epochs_run"],
            "initial_validation_loss":trace[0]["validation_mae"],
            "selected_validation_loss":chosen["validation_mae"],
            "validation_anchor_floor_hits":chosen["validation_anchor_floor_hits"],
            **{key:saved[key] for key in ("readout_parameter_distance","readout_norm","delta_norm",
                "trainable_parameter_count","n_source_stations","n_validation_stations","n_source_query",
                "n_validation_query")}})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--bootstrap-draws",type=int,default=5000)
    args=parser.parse_args()
    if args.bootstrap_draws<1:
        raise ValueError("Bootstrap draws must be positive")
    panel,thresholds,sources,states,controls=load_panel(args.root)
    choices,mixing,gamma_scores,adapter_scores=selection_records(states,panel)
    training=training_records(states)
    trace_rows, loss_rows = [], []
    for run in states:
        identity = {"split_seed":run["split_seed"],"seed":run["seed"]}
        for row in run["readout"]["trace"]:
            trace_rows.append({**identity,**{k:v for k,v in row.items() if k!="validation_by_k_ridge"}})
            loss_rows.extend({**identity,"epoch":row["epoch"],**v} for v in row["validation_by_k_ridge"])
    traces, validation_losses = pd.DataFrame(trace_rows), pd.DataFrame(loss_rows)
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
        ("adapter_choices",choices),("adapter_scores",adapter_scores),("mixing_choices",mixing),("gamma_scores",gamma_scores),("parent_replication",controls),("readout_checkpoint_choices",training),
        ("readout_training_trace",traces),("readout_validation_components",validation_losses))
    for name,frame in outputs:
        frame.to_csv(out/f"{name}.csv",index=False)
    write_report(out,curves,profiles,classification,effects,choices,mixing,controls,training,args.bootstrap_draws)
    hashes={f"{name}.csv":sha256(out/f"{name}.csv") for name,_ in outputs}
    hashes["findings.md"]=sha256(out/"findings.md")
    for name in ("analyze_doc_daily_hydro_fallback_v1.py","analyze_doc_encoder_residual_v1.py",
                 "analyze_doc_selective_residual_v1.py","analyze_doc_tail_residual_v1.py",
                 "analyze_unified_doc_spatial.py","analyze_unified_doc_spatial_v2.py","analyze_unified_doc_spatial_v3.py"):
        path=Path(__file__).parent/name;sources.append({"path":str(path),"sha256":sha256(path)})
    (out/"analysis_manifest.json").write_text(json.dumps({
        "analysis_script":str(Path(__file__)),"analysis_script_sha256":sha256(Path(__file__)),
        "bootstrap_draws":args.bootstrap_draws,"models":MODELS,"k_values":KS,"comparison_count":len(definitions),
        "sources":sources,"outputs":hashes,"role":"same-cohort source-trained readout development; no target model selection",
        "estimand":"cell-pooled within seed; seed mean within partition; equal partition mean"},indent=2)+"\n")
    print(curves[curves.k.isin((0,5))][["model_name","k","mae","rmse"]].to_string(index=False))
    print(f"Saved all 8 support-readout models and {len(definitions)} fixed contrasts to {out}")


if __name__=="__main__":
    main()
