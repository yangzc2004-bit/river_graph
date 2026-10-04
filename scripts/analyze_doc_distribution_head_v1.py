"""Evaluate point and conditional-density DOC heads on fixed station holdouts.

All heads use the same query cells and source-validation selection protocol.
Point performance, not density fit or interval coverage, determines interpretation.
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

ROOT = Path("experiments/phase4_transfer/doc_distribution_head_v1")
ARMS = ("point", "single", "mixture")
SHAPES = ("constant", "gru_tuned_anchor")
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS
INTERACTIONS = [0, 2, 4, 28, 30, 31, 32]


def comparison_definitions():
    return [(f"{candidate}_vs_{reference}{suffix}_gru_tuned_anchor_k{k}",
             f"{candidate}{suffix}_gru_tuned_anchor",k,f"{reference}{suffix}_gru_tuned_anchor",k,
             ("density_mechanism" if reference=="single" else "density_vs_point")+
             ("_integrated" if suffix else "_direct"))
            for suffix in ("", "_integrated")
            for candidate,reference in (("single","point"),("mixture","point"),("mixture","single"))
            for k in (0,5)]


def bound_file(run, name, completion, sources):
    path=run/name
    value=sha256(path)
    if completion['files'].get(name)!=value:
        raise ValueError(f'Unbound or changed artifact: {path}')
    sources.append({'path':str(path),'sha256':value})
    return path


def load_panel(root):
    frames,thresholds,sources,states,controls=[],{},[],[],[]
    daily_identity=None
    for split in SPLITS:
        for seed in SEEDS:
            run=root/'runs'/f'split{split}_seed{seed}'
            frame,config,completion=read_predictions(run,sources)
            if ((config['split_seed'],config['seed'])!=(split,seed)
                    or config['experiment']!='doc_distribution_head_v1'
                    or config['arms']!=list(ARMS) or tuple(config['basis_names'])!=SHAPES
                    or set(config['models'])!=set(MODELS) or tuple(config['k_values'])!=KS
                    or config['inference_roles']!=['train'] or config['selection_role']!='source_validation'
                    or config['target_analyte']!='doc' or config['target_transform']!='log1p'
                    or config['backbone_retraining'] is not False or config['forest_retraining'] is not False
                    or config['readout_fitting'] is not False or config['feature_dim']!=550
                    or config['feature_batch_size']!=512 or config['epochs']!=100 or config['patience']!=10
                    or config['batch_size']!=512 or config['learning_rate']!=.001 or config['gradient_clip_norm']!=1
                    or config['extra_dim']!=38 or config['interaction_indices']!=INTERACTIONS
                    or config['correction_scales']!=[0,.25,.5,1] or config['sigma_floor']!=.03
                    or config['sigma_initial_floor']!=.05 or config['feature_std_floor']!=1e-6
                    or config['median_iterations']!=64 or config['median_bracket_sigma']!=12
                    or config['head_parameters']!={'single':1102,'mixture':2755}
                    or config['support_basis']!='unchanged v4 GRU and constant'):
                raise ValueError('Unexpected distribution-head experiment settings')
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
                'daily_features_hash','daily_metadata_hash','inference_roles','feature_definition'))
            if old_config['experiment']!='doc_daily_hydro_memory_v1':
                raise ValueError('Point parent is not the retained memory-study expert')
            parent_checkpoint=bound_file(prior,'off.pt',old_completion,sources)
            if sha256(parent_checkpoint)!=config['parent_checkpoint_hash']:
                raise ValueError('Frozen point checkpoint changed')
            features=read_bound_json(run,'feature_definition.json',completion,sources)
            if (features!=config['feature_definition'] or len(features['feature_names'])!=38
                    or features['combined_interaction_indices']!=INTERACTIONS
                    or features['daily_feature_names']!=metadata['feature_names']):
                raise ValueError('Changed underlying expert head features')
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
            for field in ('context_pred','ecological_memory'):
                np.testing.assert_array_equal(full[field],old_full[field])
            for new,old in (('point_pred','off_pred'),('point_integrated_k0_pred','off_integrated_k0_pred')):
                np.testing.assert_array_equal(full[new],old_full[old])
            for arm,count in (('single',1),('mixture',2)):
                weights=full[[f'{arm}_weights_{i}' for i in range(count)]].to_numpy()
                means=full[[f'{arm}_means_{i}' for i in range(count)]].to_numpy()
                scales=full[[f'{arm}_scales_{i}' for i in range(count)]].to_numpy()
                median=full[f'{arm}_median_residual'].to_numpy()
                if (not all(np.isfinite(value).all() for value in (weights,means,scales,median))
                        or (weights<0).any() or (weights>1).any() or (scales<.03).any()):
                    raise ValueError('Nonfinite/invalid conditional density parameters')
                np.testing.assert_allclose(weights.sum(1),1,rtol=0,atol=1e-14)
                if count==1:
                    np.testing.assert_array_equal(median,means[:,0])
                elif (means[:,1]<means[:,0]).any():
                    raise ValueError('Mixture components are not ordered as defined')
            adapters=read_bound_json(run,'adapters.json',completion,sources)
            mixers=read_bound_json(run,'mixers.json',completion,sources)
            old_adapters=read_bound_json(prior,'adapters.json',old_completion,sources)
            old_mixers=read_bound_json(prior,'mixers.json',old_completion,sources)
            mapping={f'context_{shape}':f'context_{shape}' for shape in SHAPES}
            mapping.update({f'point{suffix}_{shape}':f'off{suffix}_{shape}'
                            for suffix in ('','_integrated') for shape in SHAPES})
            for name,old_name in mapping.items():
                a=frame[frame.model_name.eq(name)].sort_values(['k','cell'])
                b=previous[previous.model_name.eq(old_name)].sort_values(['k','cell'])
                for field in ('cell','k','y_true','y_pred'):
                    np.testing.assert_array_equal(a[field],b[field])
                fits,old_fits=(mixers,old_mixers) if '_integrated_' in name else (adapters,old_adapters)
                if fits[name]!=old_fits[old_name]:
                    raise ValueError('Copied context/point support selection differs from parent')
                controls.append({'split_seed':split,'seed':seed,'model_name':name,'parent_model':old_name,
                    'control_role':'copied_context_or_point','n_query_rows':len(a),'bitwise_exact':True})
            for base in ('context',*ARMS,*(f'{arm}_integrated' for arm in ARMS)):
                a=frame[frame.model_name.eq(f'{base}_constant')&frame.k.eq(0)].sort_values('cell')
                b=frame[frame.model_name.eq(f'{base}_gru_tuned_anchor')&frame.k.eq(0)].sort_values('cell')
                np.testing.assert_array_equal(a.cell,b.cell)
                np.testing.assert_array_equal(a.y_pred,b.y_pred)
                field=f'{base}_k0_pred' if '_integrated' in base else f'{base}_pred'
                np.testing.assert_array_equal(a.y_pred,full.loc[a.cell,field])
            checks=read_bound_json(run,'point_checks.json',completion,sources)
            if len(checks)!=16 or not all(v['bitwise_exact'] for v in checks):
                raise ValueError('Incomplete point-control runner checks')
            training={arm:read_bound_json(run,f'{arm}.json',completion,sources) for arm in ARMS[1:]}
            for arm,state in training.items():
                if state['selected_scale']==0:
                    np.testing.assert_array_equal(full[f'{arm}_pred'],full['point_pred'])
                    for suffix in ('','_integrated'):
                        for shape in SHAPES:
                            name,reference=f'{arm}{suffix}_{shape}',f'point{suffix}_{shape}'
                            a=frame[frame.model_name.eq(name)].sort_values(['k','cell'])
                            b=frame[frame.model_name.eq(reference)].sort_values(['k','cell'])
                            np.testing.assert_array_equal(a.y_pred,b.y_pred)
                            fits=mixers if suffix else adapters
                            if fits[name]!=fits[reference]:
                                raise ValueError('Zero density blend changed support or ecological calibration')
            inputs=read_bound_json(run,'input_definition.json',completion,sources)
            source_path=bound_file(run,'source_training.npz',completion,sources)
            if (inputs['feature_dim']!=550 or inputs['source_feature_shape'][1]!=550
                    or inputs['validation_feature_shape'][1]!=550 or inputs['source_neural_is_oof'] is not False
                    or inputs['source_visibility']!='receiving station-fold hidden' or inputs['full_visibility']!='train DOC only'):
                raise ValueError('Changed frozen density source-feature definition')
            with np.load(source_path,allow_pickle=False) as archive:
                for name in ('source_base','validation_base'):
                    if not np.isfinite(archive[name]).all() or (archive[name]<0).any():
                        raise ValueError('Invalid density source/validation native base')
                if (len(archive['source_cells'])!=inputs['source_feature_shape'][0]
                        or len(archive['validation_cells'])!=inputs['validation_feature_shape'][0]):
                    raise ValueError('Density source/validation feature counts differ')
                np.testing.assert_array_equal(archive['source_station_ids'],inputs['source_station_ids'])
                np.testing.assert_array_equal(archive['validation_base'],full.loc[archive['validation_cells'],'point_pred'])
            bound_file(run,'source_validation.csv',completion,sources)
            for arm in ARMS[1:]:
                bound_file(run,f'{arm}_trace.csv',completion,sources)
            states.append({'split_seed':split,'seed':seed,'config':config,'adapters':adapters,
                           'mixers':mixers,'training':training,'inputs':inputs})
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
        for arm,state in run['training'].items():
            cfg,definition,trace=state['config'],state['definition'],state['trace']
            count=1 if arm=='single' else 2
            if (state['model_class']!='DistributionalResidualHead' or state['schema_version']!=1
                    or cfg['n_features']!=550 or cfg['components']!=count or cfg['epochs']!=100
                    or cfg['patience']!=10 or cfg['batch_size']!=512 or cfg['learning_rate']!=.001
                    or cfg['seed']!=run['seed'] or state['selection_role']!='source_validation'
                    or state['trainable_parameter_count']!=(1102 if count==1 else 2755)
                    or definition['objective']!='unweighted_source_gaussian_residual_nll'
                    or definition['point']!='mixture_median' or definition['median_iterations']!=64
                    or definition['sigma_floor']!=.03 or definition['sigma_upper_cap'] is not None
                    or definition['correction_scales']!=[0,.25,.5,1]
                    or definition['selection']!='pooled_source_validation_native_mae'
                    or definition['normalization_role']!='source_training' or definition['feature_std_floor']!=1e-6):
                raise ValueError('Unexpected conditional-density head training definition')
            if [row['epoch'] for row in trace]!=list(range(state['epochs_run']+1)):
                raise ValueError('Incomplete density-head training trace')
            chosen=min(trace,key=lambda row:(row['validation_mae'],row['selected_scale'],row['epoch']))
            best_nll=min(trace,key=lambda row:(row['validation_nll'],row['epoch']))
            selected_keys=('validation_mae','selected_scale','validation_nll','scale_scores')
            if (chosen['epoch']!=state['best_epoch'] or chosen['selected_scale']!=state['selected_scale']
                    or {key:chosen[key] for key in selected_keys}!=state['validation_metrics']
                    or state['initial_validation_mae']!=trace[0]['validation_mae']
                    or not np.isfinite([[row[key] for key in ('training_nll','validation_nll','validation_mae')]
                                        for row in trace]).all()):
                raise ValueError('Density checkpoint differs from source-validation MAE selection')
            for row in trace:
                scores=row['scale_scores']
                if [score['scale'] for score in scores]!=[0,.25,.5,1]:
                    raise ValueError('Changed density-median blend grid')
                winner=min(scores,key=lambda value:(value['mae'],value['scale']))
                if (winner['mae']!=row['validation_mae'] or winner['scale']!=row['selected_scale']
                        or scores[0]['mae']!=state['baseline_validation_mae']):
                    raise ValueError('Density blend selection differs from recorded candidate losses')
            norm=state['normalization']
            mean,std,scale=[np.asarray(norm[key]) for key in ('feature_mean','feature_raw_std','feature_scale')]
            if (any(value.shape!=(550,) or not np.isfinite(value).all() for value in (mean,std,scale))
                    or (std<0).any() or (scale<=0).any()
                    or norm['unit_scale_feature_count']!=int((std<1e-6).sum())):
                raise ValueError('Invalid source-fitted feature normalization')
            np.testing.assert_array_equal(scale,np.where(std<1e-6,1.,std))
            rows.append({'split_seed':run['split_seed'],'seed':run['seed'],'arm':arm,
                'components':count,'trainable_parameter_count':state['trainable_parameter_count'],
                'n_source_cells':state['n_source_cells'],'n_validation_query':state['n_validation_query'],
                'best_epoch':state['best_epoch'],'epochs_run':state['epochs_run'],'selected_scale':state['selected_scale'],
                'exact_point_fallback':state['selected_scale']==0,
                'baseline_validation_mae':state['baseline_validation_mae'],
                'initial_validation_mae':state['initial_validation_mae'],'selected_validation_mae':chosen['validation_mae'],
                'initial_validation_nll':trace[0]['validation_nll'],'selected_validation_nll':chosen['validation_nll'],
                'minimum_validation_nll':best_nll['validation_nll'],'minimum_validation_nll_epoch':best_nll['epoch'],
                'selected_training_nll_trace':chosen['training_nll'],
                'selected_source_nll':state['selected_source_nll'],
                'unit_scale_feature_count':norm['unit_scale_feature_count']})
    return pd.DataFrame(rows)


def density_diagnostics(states):
    traces,scale_rows,parameters,structure=[],[],[],[]
    for run in states:
        for arm,state in run['training'].items():
            identity={'split_seed':run['split_seed'],'seed':run['seed'],'arm':arm}
            for row in state['trace']:
                traces.append({**identity,**{key:value for key,value in row.items() if key!='scale_scores'}})
                scale_rows.extend({**identity,'epoch':row['epoch'],**value} for value in row['scale_scores'])
            for cohort in ('source','validation'):
                block=state[f'{cohort}_distribution']
                if block['n_rows']!=state['n_source_cells' if cohort=='source' else 'n_validation_query']:
                    raise ValueError('Density diagnostic population differs from fit/selection cohort')
                for kind in ('means','scales','weights'):
                    if len(block[kind])!=state['config']['components']:
                        raise ValueError('Density diagnostic component count differs')
                    for component,values in enumerate(block[kind]):
                        if not np.isfinite(list(values.values())).all():
                            raise ValueError('Nonfinite source density summary')
                        parameters.append({**identity,'cohort':cohort,'component':component,
                            'parameter':kind,'n_rows':block['n_rows'],**values})
                row={**identity,'cohort':cohort,'n_rows':block['n_rows'],
                     'sigma_near_floor_fraction':block['sigma_near_floor_fraction']}
                if state['config']['components']==2:
                    row.update({key:block[key] for key in ('mean_gap_below_0_01_fraction',
                        'high_weight_below_0_01_fraction','high_weight_above_0_99_fraction')})
                    row.update({f'mean_gap_{key}':value for key,value in block['mean_gap'].items()})
                if any(not 0<=value<=1 for key,value in row.items() if key.endswith('_fraction')):
                    raise ValueError('Invalid density concentration fraction')
                structure.append(row)
    return tuple(pd.DataFrame(rows) for rows in (traces,scale_rows,parameters,structure))


def write_report(out,curves,profiles,classification,effects,training,controls,draws):
    lines=["# Conditional distribution heads for sparse DOC reconstruction","",
        "Three heads are compared: the retained point model, a single conditional distribution,",
        "and a conditional mixture. Density models provide median point predictions. Forests,",
        "ecological residual profiles, support representations, and query cells remain unchanged.",
        "The density models are fitted on source training data; source-validation native MAE",
        "selects their checkpoint and blend, including the exact zero-blend fallback. A better",
        "likelihood fit alone does not establish a better reconstruction model.","",
        "All fourteen model curves and twelve specified contrasts are reported. Direct comparisons",
        "include the source-validation support calibration; integrated comparisons additionally",
        "include source-validation-selected ecological mixing. Constant-only support is retained",
        "as a diagnostic. The principal contrasts use the unchanged GRU support basis at K0/K5.","",
        f"All intervals use {draws:,} paired whole-station bootstrap draws. Repeated station identities",
        "are resampled jointly across partitions. Seed means average within each partition, then",
        "partitions receive equal weight. Q90 includes ties at the source-training threshold.",
        "These are reused development partitions; target outcomes do not choose a head, blend,",
        "checkpoint, support calibration, ecological mixture or K-specific route.","",
        f"The loader verifies {len(controls)} copied-context/point model comparisons.","",
        "## Complete K curves","",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values=curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | "+" | ".join(f"{values.loc[k,'mae']:.6f}" for k in KS)
                     +f" | {values.loc[5,'rmse']:.6f} | {values.loc[5,'r2']:.6f} |")
    for role,title in (("density_vs_point_direct","Density medians versus point head: direct"),
                       ("density_vs_point_integrated","Density medians versus point head: integrated"),
                       ("density_mechanism_direct","Mixture versus single distribution: direct"),
                       ("density_mechanism_integrated","Mixture versus single distribution: integrated")):
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
    lines += ["","## Tail and ordinary DOC with detection tradeoffs","",
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
    lines += ["","## Source-validation fitting and model selection","",
        "Epoch zero uses a source-initialized constant density; it is not necessarily the exact",
        "point fallback. Only correction scale zero guarantees unchanged native point predictions.",
        "All feature normalization and density initialization statistics use source training rows.",
        "The source base includes a source-trained neural correction over forest OOF predictions;",
        "the entire neural source base is not claimed to be OOF.","",
        "Saved source-training and source-validation likelihood traces are separate from native-MAE",
        "checkpoint/blend choices. MAE ties prefer the lower correction scale, then the earlier epoch.",
        "Epoch>0 training-NLL traces average online minibatch losses; selected_source_nll evaluates",
        "the complete source set at the selected checkpoint. Component collapse and scale diagnostics describe the conditional",
        "density fit. They are not additional target endpoints or evidence of calibrated uncertainty.",
        "Any density intervals are exploratory distribution diagnostics; this study promotes models",
        "only through point-prediction performance. All selected blending and support/mixing parameters",
        "are preserved with their source-validation candidate scores.","",
        "| Arm | Selected epochs | Exact point fallbacks | Mean baseline MAE | Mean selected MAE | Mean initial NLL | Mean selected NLL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,group in training.groupby('arm',sort=False):
        lines.append(f"| {arm} | {group.best_epoch.min()}–{group.best_epoch.max()} | "
            f"{int(group.exact_point_fallback.sum())}/{len(group)} | {group.baseline_validation_mae.mean():.6f} | "
            f"{group.selected_validation_mae.mean():.6f} | {group.initial_validation_nll.mean():.6f} | "
            f"{group.selected_validation_nll.mean():.6f} |")
    lines += ["",
        "Partition/seed directions and station gain/harm concentration accompany aggregate effects.",
        "Intervals crossing zero do not establish equivalence. No target-based route is produced.",""]
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
    training=training_records(states)
    traces,scale_scores,distribution_parameters,density_structure=density_diagnostics(states)
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
        ("parent_replication",controls),("training_choices",training),
        ("density_training_traces",traces),("density_scale_scores",scale_scores),
        ("source_density_parameters",distribution_parameters),("source_density_structure",density_structure))
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
        "sources":sources,"outputs":hashes,
        "role":"same-cohort conditional-density-head development; no target model selection",
        "estimand":"cell-pooled within seed; seed mean within partition; equal partition mean"},indent=2)+"\n")
    print(curves[curves.k.isin((0,5))][["model_name","k","mae","rmse"]].to_string(index=False))
    print(f"Saved all 14 distribution-head models and {len(definitions)} fixed contrasts to {out}")


if __name__=="__main__":
    main()
