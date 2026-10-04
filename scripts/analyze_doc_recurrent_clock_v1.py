"""Evaluate recurrent missingness clocks in the existing DOC residual model.

All arms retain the same target queries and source-selected support protocol.
This analyzer reports every planned clock contrast without target-based routing.
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

ROOT = Path("experiments/phase4_transfer/doc_recurrent_clock_v1")
ARMS = ("legacy", "unseen_neutral", "flow_window")
SHAPES = ("constant", "gru_tuned_anchor")
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS
INTERACTIONS = [0, 2, 4, 28, 30, 31, 32]


def comparison_definitions():
    return [(f"{arm}_vs_legacy{suffix}_gru_tuned_anchor_k{k}",
             f"{arm}{suffix}_gru_tuned_anchor",k,f"legacy{suffix}_gru_tuned_anchor",k,
             "recurrent_clock_direct" if not suffix else "recurrent_clock_integrated")
            for suffix in ("", "_integrated") for arm in ("unseen_neutral", "flow_window") for k in (0,5)]


def bound_file(run, name, completion, sources):
    path=run/name
    value=sha256(path)
    if completion['files'].get(name)!=value:
        raise ValueError(f'Unbound or changed artifact: {path}')
    sources.append({'path':str(path),'sha256':value})
    return path


def load_panel(root, expected_epochs=120):
    frames,thresholds,sources,states,controls=[],{},[],[],[]
    daily_identity=None
    for split in SPLITS:
        for seed in SEEDS:
            run=root/'runs'/f'split{split}_seed{seed}'
            frame,config,completion=read_predictions(run,sources)
            if ((config['split_seed'],config['seed'])!=(split,seed)
                    or config['experiment']!='doc_recurrent_clock_v1'
                    or config['arms']!=list(ARMS) or tuple(config['basis_names'])!=SHAPES
                    or set(config['models'])!=set(MODELS) or tuple(config['k_values'])!=KS
                    or config['inference_roles']!=['train'] or config['selection_role']!='source_validation'
                    or config['target_analyte']!='doc' or config['target_transform']!='log1p'
                    or config['forest_retraining'] is not False or config['readout_fitting'] is not False
                    or config['epochs']!=expected_epochs or config['patience']!=5
                    or config['lookback']!=12 or config['batch_size']!=512
                    or config['learning_rate']!=1e-4 or config['head_learning_rate']!=1e-3
                    or config['encoder_learning_rate']!=1e-5 or config['encoder_mode']!='last_self_ecology'
                    or config['hydro_sequence_mode']!='off' or config['extra_dim']!=38
                    or config['interaction_indices']!=INTERACTIONS or config['tail_weight']!=2
                    or config['residual_scales']!=[0,.25,.5,1] or config['train_memory'] is not True
                    or config['new_neural_fits']!=2 or config['support_basis']!='unchanged frozen v4 GRU in all arms'):
                raise ValueError('Unexpected recurrent-clock experiment settings')
            threshold=float(config['q90_threshold_train'])
            if not np.isfinite(threshold) or threshold<0:
                raise ValueError('Invalid source-training Q90')
            thresholds[(split,seed)]=threshold
            for name in ('prior','source','basis','oof'):
                path=Path(config[f'{name}_run'])/'complete.json'
                if sha256(path)!=config[f'{name}_completion_hash']:
                    raise ValueError(f'Changed frozen {name} package')
                sources.append({'path':str(path),'sha256':config[f'{name}_completion_hash']})
            paths=(Path(config['daily_features_path']),Path(config['daily_metadata_path']))
            hashes=(config['daily_features_hash'],config['daily_metadata_hash'])
            if any(sha256(path)!=value for path,value in zip(paths,hashes,strict=True)):
                raise ValueError('Changed daily-feature input')
            if daily_identity is not None and hashes!=daily_identity:
                raise ValueError('Daily input changed between runs')
            if daily_identity is None:
                daily_identity=hashes
                sources.extend({'path':str(path),'sha256':value} for path,value in zip(paths,hashes,strict=True))
            metadata=json.loads(paths[1].read_text())
            if metadata['dataset_hash']!=config['dataset_hash']:
                raise ValueError('Daily inputs have different dataset identity')
            prior=Path(config['prior_run'])
            previous,old_config,old_completion=read_predictions(prior,sources)
            check_source_identity(config,old_config,('split_seed','seed','dataset_hash','mask_hash',
                'source_run','source_completion_hash','basis_run','basis_completion_hash',
                'oof_run','oof_completion_hash','q90_threshold_train','query_cells',
                'daily_features_hash','daily_metadata_hash','inference_roles','feature_definition','patience'))
            if old_config['experiment']!='doc_daily_hydro_memory_v1' or old_config['epochs']!=expected_epochs:
                raise ValueError('Legacy control requires the completed parent with the same epoch ceiling')
            parent_checkpoint=bound_file(prior,'off.pt',old_completion,sources)
            if (sha256(parent_checkpoint)!=config['parent_checkpoint_hash']
                    or sha256(run/'legacy.pt')!=config['parent_checkpoint_hash']):
                raise ValueError('Legacy checkpoint no longer equals the selected parent off')
            features=read_bound_json(run,'feature_definition.json',completion,sources)
            if (features!=config['feature_definition'] or len(features['feature_names'])!=38
                    or features['combined_interaction_indices']!=INTERACTIONS
                    or features['daily_feature_names']!=metadata['feature_names']):
                raise ValueError('Changed head feature definition')
            if sha256(run/'feature_definition.json')!=sha256(prior/'feature_definition.json'):
                raise ValueError('Feature definition failed byte-exact parent copy')
            basis_run=Path(config['basis_run'])
            basis_completion=json.loads((basis_run/'complete.json').read_text())
            bound_file(basis_run,'representations.npz',basis_completion,sources)
            full=read_full_grid(run,config,completion,sources)
            old_full=read_full_grid(prior,old_config,old_completion,sources)
            if sha256(prior/'full_grid.parquet')!=config['parent_full_grid_hash']:
                raise ValueError('Changed parent native component product')
            np.testing.assert_array_equal(full.index,old_full.index)
            for field in ('context_pred','ecological_memory',
                          'observation_age_months','visible_target_history','daily_numeric_valid_count',
                          'daily_history_valid_months','daily_history_possible_months'):
                np.testing.assert_array_equal(full[field],old_full[field])
            for new,old in (('legacy_pred','off_pred'),('legacy_delta','off_delta'),
                            ('legacy_integrated_k0_pred','off_integrated_k0_pred')):
                np.testing.assert_array_equal(full[new],old_full[old])
            adapters=read_bound_json(run,'adapters.json',completion,sources)
            mixers=read_bound_json(run,'mixers.json',completion,sources)
            old_adapters=read_bound_json(prior,'adapters.json',old_completion,sources)
            old_mixers=read_bound_json(prior,'mixers.json',old_completion,sources)
            mapping={f'context_{shape}':f'context_{shape}' for shape in SHAPES}
            mapping.update({f'legacy{suffix}_{shape}':f'off{suffix}_{shape}'
                            for suffix in ('','_integrated') for shape in SHAPES})
            for name,old_name in mapping.items():
                a=frame[frame.model_name.eq(name)].sort_values(['k','cell'])
                b=previous[previous.model_name.eq(old_name)].sort_values(['k','cell'])
                for field in ('cell','k','y_true','y_pred'):
                    np.testing.assert_array_equal(a[field],b[field])
                fits,old_fits=(mixers,old_mixers) if '_integrated_' in name else (adapters,old_adapters)
                if fits[name]!=old_fits[old_name]:
                    raise ValueError('Copied context/legacy support selection differs from parent')
                controls.append({'split_seed':split,'seed':seed,'model_name':name,'parent_model':old_name,
                    'control_role':'copied_context_or_legacy','n_query_rows':len(a),'bitwise_exact':True})
            for base in ('context',*ARMS,*(f'{arm}_integrated' for arm in ARMS)):
                a=frame[frame.model_name.eq(f'{base}_constant')&frame.k.eq(0)].sort_values('cell')
                b=frame[frame.model_name.eq(f'{base}_gru_tuned_anchor')&frame.k.eq(0)].sort_values('cell')
                np.testing.assert_array_equal(a.cell,b.cell)
                np.testing.assert_array_equal(a.y_pred,b.y_pred)
                field=f'{base}_k0_pred' if '_integrated' in base else f'{base}_pred'
                np.testing.assert_array_equal(a.y_pred,full.loc[a.cell,field])
            checks=read_bound_json(run,'legacy_checks.json',completion,sources)
            if len(checks)!=16 or not all(v['bitwise_exact'] for v in checks):
                raise ValueError('Incomplete legacy runner checks')
            training={arm:read_bound_json(run,f'{arm}.json',completion,sources) for arm in ARMS}
            if training['legacy']!=read_bound_json(prior,'off.json',old_completion,sources):
                raise ValueError('Reused legacy training summary changed')
            clocks=read_bound_json(run,'clock_diagnostics.json',completion,sources)
            if set(clocks)!=set(ARMS):
                raise ValueError('Incomplete source-validation clock diagnostics')
            sample=None
            for arm,state in clocks.items():
                cells=np.asarray(state['sample_cells'])
                if sample is None:
                    sample=cells
                np.testing.assert_array_equal(cells,sample)
                if (cells.ndim!=1 or not len(cells) or len(cells)>2048
                        or len(np.unique(cells))!=len(cells) or state['valid_steps']<len(cells)
                        or state['valid_steps']>len(cells)*12
                        or not np.isfinite([state['clock_mean'],state['mean_gamma'],*state['clock_quantiles']]).all()
                        or not 0<=state['mean_gamma']<=1):
                    raise ValueError('Invalid sampled source-validation clock summary')
                if arm=='unseen_neutral' and (state['clock_mean']!=0 or any(state['clock_quantiles'])):
                    raise ValueError('Source-validation held-station neutral clock is not zero')
                if arm=='flow_window' and (state['clock_mean']<0 or max(state['clock_quantiles'])>1):
                    raise ValueError('Flow clock exceeds the specified twelve-month normalized range')
            bound_file(run,'source_validation.csv',completion,sources)
            for arm in ARMS[1:]:
                bound_file(run,f'{arm}_trace.csv',completion,sources)
            states.append({'split_seed':split,'seed':seed,'config':config,'adapters':adapters,
                           'mixers':mixers,'training':training,'clocks':clocks})
            frames.append(frame)
        if len({thresholds[(split,seed)] for seed in SEEDS})!=1:
            raise ValueError('Source Q90 differs across training seeds')
    panel=pd.concat(frames,ignore_index=True)
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


def training_records(states):
    rows=[]
    for run in states:
        counts=set()
        for arm,state in run['training'].items():
            cfg,trace=state['config'],state['trace']
            expected_class='EncoderNativeResidual' if arm=='legacy' else 'ClockNativeResidual'
            if (state['model_class']!=expected_class or cfg.get('decay_clock','legacy')!=arm
                    or cfg['tail_weight']!=2 or cfg['encoder_mode']!='last_self_ecology'
                    or cfg['encoder_learning_rate']!=1e-5 or cfg['extra_dim']!=38
                    or cfg['interaction_indices']!=INTERACTIONS
                    or cfg['learning_rate']!=1e-4 or cfg['head_learning_rate']!=1e-3
                    or cfg.get('hydro_sequence_mode','off')!='off'
                    or cfg['lookback']!=12 or cfg['batch_size']!=512
                    or cfg['scales']!=[0,.25,.5,1] or cfg['seed']!=run['seed']
                    or cfg['epochs']!=run['config']['epochs'] or cfg['patience']!=5
                    or not cfg.get('train_memory',True)
                    or state['protocol']['selection_role']!='source_validation'):
                raise ValueError('Unexpected saved recurrent-clock training definition')
            if arm!='legacy':
                protocol=state['protocol']
                if (protocol['decay_clock']!=arm or protocol['flow_visibility_raw_index']!=3
                        or protocol['last_observation_valid_raw_index']!=-7
                        or protocol['clock_input_changed']!='decay scalar only; raw M1 age unchanged'):
                    raise ValueError('Changed recurrent-clock observation convention')
            if [v['epoch'] for v in trace]!=list(range(state['epochs_run']+1)):
                raise ValueError('Incomplete recurrent-clock training trace')
            chosen=min(trace,key=lambda row:(row['validation_mae'],row['epoch']))
            if (chosen['epoch']!=state['best_epoch'] or chosen['validation_scale']!=state['selected_scale']
                    or not np.isfinite([row['validation_mae'] for row in trace]).all()):
                raise ValueError('Checkpoint selection differs from source-validation trace')
            for row in trace:
                winner=min(row['validation_candidates'],key=lambda v:(v['mae'],v['scale']))
                if winner['mae']!=row['validation_mae'] or winner['scale']!=row['validation_scale']:
                    raise ValueError('Scale selection differs from source-validation candidates')
            counts.add(state['trainable_parameter_count'])
            rows.append({'split_seed':run['split_seed'],'seed':run['seed'],'arm':arm,
                'epoch_ceiling':cfg['epochs'],'best_epoch':state['best_epoch'],'epochs_run':state['epochs_run'],
                'selected_scale':state['selected_scale'],'initial_validation_mae':trace[0]['validation_mae'],
                'selected_validation_mae':chosen['validation_mae'],
                **{key:state[key] for key in ('trainable_parameter_count','n_source_cells','n_source_tail_cells',
                    'n_validation_query','temporal_parameter_distance','decay_parameter_distance','head_parameter_norm',
                    'encoder_trainable_parameter_count','spatial_parameter_distance','last_self_parameter_distance',
                    'ecology_parameter_distance')}})
        if len(counts)!=1:
            raise ValueError('Clock comparison changed allocated trainable parameter count')
    return pd.DataFrame(rows)


def write_report(out, curves, profiles, classification, effects, training, controls, draws):
    lines = ["# Recurrent clock comparison for spatial DOC transfer", "",
        "The encoder/head architecture, inputs, original initialization, trainable scope, loss and",
        "budget are held constant. Each new clock refits the last spatial self layer, ecology encoder,",
        "GRU, decay layer and native residual head. Forests, the ecological profile and support-basis",
        "weights remain fixed. The changed design input is the scalar supplied to recurrent decay.",
        "Legacy is the completed parent off expert;",
        "unseen-neutral and flow-window are two new matched-budget fits. No forest is retrained.", "",
        "Unseen-neutral sets only the decay-layer age scalar to zero until a local DOC observation",
        "has been visible. It does not disable decay or change the raw M1 age feature. Flow-window",
        "starts its clock at twelve months within each queried causal window, resets on visible",
        "monthly discharge, otherwise increments to a cap of twelve, and skips padded dates.",
        "The flow clock uses monthly discharge visibility, not daily descriptor validity. It encodes",
        "shared-state decay rather than river travel time, residence time, or a physical causal effect.", "",
        "Checkpoint and global residual scale use source-validation K0 MAE, including the exact",
        "zero-residual fallback. The same source-validation support grids are used independently",
        "for every arm. Ecological gamma is selected for each arm; integrated contrasts therefore",
        "include those mixing choices in addition to the recurrent-clock change.", "",
        "All fourteen model curves and the eight prespecified clock-versus-legacy comparisons are",
        "reported. Constant-only support adaptation remains a diagnostic; bootstrap comparisons use",
        "the unchanged GRU support basis at K0 and K5. These are reused development partitions,",
        "with no model, K, route or endpoint selected from target outcomes.", "",
        f"Intervals use {draws:,} paired whole-station bootstrap draws. Repeated station identities",
        "are jointly resampled across partitions; seed means average within partition and partitions",
        "receive equal weight. Q90 includes ties at the source-training threshold. Negative MAE/FPR",
        "deltas favor the candidate; recall is reported alongside false-positive rate.", "",
        f"The loader verifies {len(controls)} exact copied-context or legacy control comparisons.", "",
        "## Complete K curves", "",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values=curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | "+" | ".join(f"{values.loc[k,'mae']:.6f}" for k in KS)
                     +f" | {values.loc[5,'rmse']:.6f} | {values.loc[5,'r2']:.6f} |")
    for role,title in (("recurrent_clock_direct","Direct recurrent-clock contrasts"),
                       ("recurrent_clock_integrated","Integrated recurrent-clock contrasts")):
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
    lines += ["","## High and ordinary DOC with detection tradeoffs","",
        "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        for k in (0,5):
            f=profiles[profiles.model_name.eq(model)&profiles.k.eq(k)].set_index("region")
            c=classification[classification.model_name.eq(model)&classification.k.eq(k)].iloc[0]
            lines.append(f"| {model} | {k} | {f.loc['q90','mae']:.6f} | {f.loc['q90','signed_bias']:+.6f} | "
                f"{f.loc['nontail','mae']:.6f} | {f.loc['nontail','signed_bias']:+.6f} | "
                f"{100*c.q90_recall:.3f}% | {100*c.q90_precision:.3f}% | {100*c.q90_false_positive_rate:.3f}% |")
    lines += ["","## Source-validation checkpoint selection","",
        "| Arm | Selected epoch range | Epochs executed | Mean initial MAE | Mean selected MAE |",
        "|---|---:|---:|---:|---:|"]
    for arm,group in training.groupby("arm",sort=False):
        lines.append(f"| {arm} | {group.best_epoch.min()}–{group.best_epoch.max()} | {group.epochs_run.sum()} | "
                     f"{group.initial_validation_mae.mean():.6f} | {group.selected_validation_mae.mean():.6f} |")
    lines += ["","Source-validation scores describe model selection, separately from target performance.",
        "All selected scales, support alpha/ridge values, ecological gamma choices and candidate",
        "scores are retained. Partition/seed directions, transformed errors and station gain/harm",
        "concentration accompany the aggregate means. Intervals crossing zero do not establish",
        "equivalence or absence of a clock effect.",""]
    (out/"findings.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--bootstrap-draws",type=int,default=5000)
    parser.add_argument("--expected-epochs",type=int,default=120)
    args=parser.parse_args()
    if args.bootstrap_draws<1 or args.expected_epochs<1:
        raise ValueError("Bootstrap draws and epoch ceiling must be positive")
    panel,thresholds,sources,states,controls=load_panel(args.root,args.expected_epochs)
    choices,mixing,gamma_scores,adapter_scores=selection_records(states,panel)
    training=training_records(states)
    clock_rows=[]
    for run in states:
        for arm,clock in run['clocks'].items():
            clock_rows.append({'split_seed':run['split_seed'],'seed':run['seed'],'arm':arm,
                'sample_cells':len(clock['sample_cells']),'valid_steps':clock['valid_steps'],
                'clock_mean':clock['clock_mean'],'mean_gamma':clock['mean_gamma'],
                **{f'clock_q{p}':v for p,v in zip((10,50,90),clock['clock_quantiles'],strict=True)},
                **{f'retention_lag{k}':v for k,v in clock['retention'].items()}})
    clock_records=pd.DataFrame(clock_rows)
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
        ("adapter_choices",choices),("adapter_scores",adapter_scores),("mixing_choices",mixing),("gamma_scores",gamma_scores),
        ("parent_replication",controls),("training_choices",training),("source_validation_clock_diagnostics",clock_records))
    for name,frame in outputs:
        frame.to_csv(out/f"{name}.csv",index=False)
    write_report(out,curves,profiles,classification,effects,training,controls,args.bootstrap_draws)
    hashes={f"{name}.csv":sha256(out/f"{name}.csv") for name,_ in outputs}
    hashes["findings.md"]=sha256(out/"findings.md")
    for name in ("analyze_doc_daily_hydro_fallback_v1.py","analyze_doc_encoder_residual_v1.py",
                 "analyze_doc_selective_residual_v1.py","analyze_doc_tail_residual_v1.py",
                 "analyze_unified_doc_spatial.py","analyze_unified_doc_spatial_v2.py","analyze_unified_doc_spatial_v3.py"):
        path=Path(__file__).parent/name;sources.append({"path":str(path),"sha256":sha256(path)})
    (out/"analysis_manifest.json").write_text(json.dumps({
        "analysis_script":str(Path(__file__)),"analysis_script_sha256":sha256(Path(__file__)),
        "bootstrap_draws":args.bootstrap_draws,"models":MODELS,"k_values":KS,"comparison_count":len(definitions),
        "expected_epochs":args.expected_epochs,"sources":sources,"outputs":hashes,
        "role":"same-cohort recurrent-clock development; no target model selection",
        "estimand":"cell-pooled within seed; seed mean within partition; equal partition mean"},indent=2)+"\n")
    print(curves[curves.k.isin((0,5))][["model_name","k","mae","rmse"]].to_string(index=False))
    print(f"Saved all 14 recurrent-clock models and {len(definitions)} fixed contrasts to {out}")


if __name__=="__main__":
    main()
