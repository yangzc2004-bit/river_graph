"""Report all four procedures, physical support and station-CV decisions."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_geographical_confirmation_v1 import (
    additional_metrics,
    improving_region_count,
    summaries,
)
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_river_frozen_comparison_v1 import ARMS, GEOMETRY, ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unmonitored_doc import HUC4_BLOCKS

CONTRASTS = [(arm, 'current_complete') for arm in ARMS[1:]]+[
    ('structure_upstream', 'simple_upstream'), ('structure_upstream', 'matched_nonupstream')]


def attach_groups(run, frame, forms):
    with np.load(run/'message_inputs.npz', allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    config = json.loads((run/'config.json').read_text())
    t = values['true_river_values'].shape[1]
    receiver = np.searchsorted(values['receivers'], frame.cell.to_numpy()//t)
    month = frame.cell.to_numpy() % t
    valid = values['true_river_valid'][receiver, month]
    raw = values['raw_valid'][receiver, month]
    owner = values['true_river_owner'][receiver]
    distance = np.expm1(values['true_river_path'][receiver, :, 0]*np.log1p(3000))
    nearest = np.where(owner >= 0, distance, np.inf).min(1)
    observed_nearest = np.where(valid.any(-1), distance, np.inf).min(1)
    labels = forms.reindex(frame.station).cluster.to_numpy()
    status = forms.reindex(frame.station).classification_status.fillna('geometry_not_available').to_numpy(dtype=object)
    frame = frame.copy()
    frame['form'] = [f'form_{int(x)}' if np.isfinite(x) else 'unclassified' for x in labels]
    frame['form_status'] = status
    frame['support_group'] = np.where(valid.any((1, 2)), 'supported', 'unsupported')
    frame['valid_source_lags'] = valid.sum((1, 2))
    frame['raw_source_lags'] = raw.sum((1, 2))
    frame['raw_supported'] = raw.any((1, 2))
    frame['nearest_path_km'] = nearest
    frame['nearest_observed_path_km'] = observed_nearest
    frame['distance_group'] = np.select([observed_nearest <= 50, observed_nearest <= 200,
        np.isfinite(observed_nearest)], ['upstream_0_50km', 'upstream_50_200km', 'upstream_over200km'],
        default='no_observed_upstream')
    frame['tail'] = frame.y_true >= config['q90_threshold_train']
    return frame


def effect(panel, candidate, reference, draws, **extra):
    pair = paired(panel, candidate, reference)
    result = {'candidate': candidate, 'reference': reference, **extra,
        **joint_station_bootstrap(pair, draws=draws), 'improved_regions': improving_region_count(pair)}
    return result, pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    forms = pd.read_csv(GEOMETRY/'station_classes.csv', dtype={'station': str}).set_index('station')
    primary, curves, thresholds, selection, validation, support_rows, sources = [], [], {}, [], [], [], {}
    for region in HUC4_BLOCKS:
        for seed in (42, 43, 44):
            run = args.root/'runs'/f'huc4_{region}_seed{seed}'
            if not (run/'complete.json').exists():
                raise ValueError('all 15 packages required, no partial scientific report')
            config = json.loads((run/'config.json').read_text())
            frame = attach_groups(run, pd.read_parquet(run/'predictions.parquet'), forms)
            primary.append(frame)
            curves.append(pd.read_parquet(run/'support_curves.parquet'))
            thresholds[(config['split_seed'], seed)] = config['q90_threshold_train']
            group = frame[frame.model_name.eq('current_complete')]
            support_rows.append({'split_seed': int(region), 'seed': seed, 'target_huc4': region,
                'test_cells': len(group), 'test_stations': group.station.nunique(),
                'mapped_path_cells': int(np.isfinite(group.nearest_path_km).sum()),
                'raw_supported_cells': int(group.raw_supported.sum()),
                'matched_supported_cells': int(group.river_supported.sum()),
                'matched_supported_stations': group[group.river_supported].station.nunique(),
                'raw_valid_source_lags': int(group.raw_source_lags.sum()),
                'common_valid_source_lags': int(group.valid_source_lags.sum())})
            for arm in ARMS[1:]:
                state = json.loads((run/f'{arm}.json').read_text())
                selection.append({'split_seed': int(region), 'seed': seed, 'model_name': arm,
                    **{key: state[key] for key in ('selected_alpha', 'n_calibration_stations',
                        'n_calibration_cells', 'coefficient_norm')},
                    'zero_selected': state['selected_alpha'] is None, 'selection_reason': state['selection_reason']})
                for score in state['station_cv_scores']:
                    validation.append({'split_seed': int(region), 'seed': seed, 'model_name': arm,
                        **score, 'zero_reference_mae': state['station_cv_scores'][0]['validation_mae']})
            with (np.load(run/'message_inputs.npz', allow_pickle=False) as saved,
                  np.load(config['mask_path'], allow_pickle=False) as mask):
                t = saved['true_river_valid'].shape[1]
                cells = mask['val']
                r, m = np.searchsorted(saved['receivers'], cells//t), cells % t
                supported = saved['true_river_valid'][r, m].any((1, 2))
                support_rows[-1].update(calibration_cells=len(cells),
                    supported_calibration_cells=int(supported.sum()),
                    supported_calibration_stations=len(np.unique(cells[supported]//t)))
            for filename in ('predictions.parquet', 'support_curves.parquet', 'complete.json'):
                sources[str(run/filename)] = sha256_file(run/filename)
    primary, curves = pd.concat(primary, ignore_index=True), pd.concat(curves, ignore_index=True)
    output = args.root/'analysis'
    output.mkdir(exist_ok=True)
    for label, panel in (('primary', primary), ('curves', curves)):
        runs, parts, summary, stations = summaries(panel, thresholds)
        for name, table in (('run_metrics', runs), ('region_metrics', parts), ('summary', summary),
                            ('station_metrics', stations), ('q90_diagnostics', additional_metrics(panel, thresholds))):
            table.to_csv(output/f'{label}_{name}.csv', index=False)
    effects, station_effects, strata = [], [], []
    for population, panel in (('all_observed_k0', primary), ('fixed_query_curve', curves)):
        for k in sorted(panel.k.unique()):
            view = panel[panel.k.eq(k)]
            for candidate, reference in CONTRASTS:
                row, pair = effect(view, candidate, reference, args.bootstrap_draws, population=population, k=k, subset='all')
                effects.append(row)
                if population == 'all_observed_k0':
                    station = pair.groupby(['split_seed', 'station'], as_index=False).agg(
                        candidate_mae=('candidate_error', 'mean'), reference_mae=('reference_error', 'mean'),
                        n_cells=('cell', 'nunique'))
                    station['candidate'], station['reference'] = candidate, reference
                    station['delta_mae'] = station.candidate_mae-station.reference_mae
                    station_effects.append(station)
                    for subset, selected in (('q90', view[view['tail']]), ('supported', view[view.river_supported]),
                                            ('unsupported', view[~view.river_supported])):
                        if not selected.empty:
                            result, _ = effect(selected, candidate, reference, args.bootstrap_draws,
                                population=population, k=k, subset=subset)
                            effects.append(result)
                    for column in ('form', 'distance_group'):
                        for group, selected in view.groupby(column):
                            result, _ = effect(selected, candidate, reference, args.bootstrap_draws,
                                stratum=column, group=group)
                            strata.append(result)
    pd.DataFrame(effects).to_csv(output/'paired_effects.csv', index=False)
    pd.concat(station_effects, ignore_index=True).to_csv(output/'station_effects.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'form_support_effects.csv', index=False)
    pd.DataFrame(support_rows).to_csv(output/'availability_by_run.csv', index=False)
    pd.DataFrame(support_rows).groupby('target_huc4', as_index=False).mean(numeric_only=True).to_csv(
        output/'availability_by_region.csv', index=False)
    pd.DataFrame(selection).to_csv(output/'readout_selection.csv', index=False)
    pd.DataFrame(validation).to_csv(output/'station_cv_scores.csv', index=False)
    unique = primary[primary.model_name.eq('current_complete')].drop_duplicates(['cell'])
    unique.groupby(['form', 'form_status'], as_index=False).agg(n_stations=('station', 'nunique'),
        n_cells=('cell', 'size'), supported_cells=('river_supported', 'sum')).to_csv(output/'form_population.csv', index=False)
    write_json(output/'sources.json', {'analyzer_sha256': sha256_file(__file__), 'inputs': sources,
        'bootstrap_draws': args.bootstrap_draws, 'bootstrap_cluster': 'station, all its months and training seeds together',
        'estimand': 'seed-mean cell loss, then equal HUC4 means; subgroup means use represented HUC4 only',
        'n_unique_test_cells': len(unique), 'n_unique_test_stations': unique.station.nunique(),
        'n_common_supported_cells': int(unique.river_supported.sum()),
        'evaluation_role': 'retrospective ST357 geographical withholding, not external validation',
        'calibration_role': 'held-gradient base-validation stations; previously used for base selection, not complete-model OOF'})
    print(pd.read_csv(output/'primary_summary.csv')[['model_name', 'mae', 'q90_mae', 'station_equal_mae']].to_string(index=False))


if __name__ == '__main__':
    main()
