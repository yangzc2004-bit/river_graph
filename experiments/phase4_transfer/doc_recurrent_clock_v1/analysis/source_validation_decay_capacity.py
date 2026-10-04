"""Inspect selected neutral-clock decay using source-validation inputs only.

No DOC query labels, target predictions, model fitting, or model selection is
used. The calculation replays the selected decay tensors on every valid step
of every fixed source-validation K0 query window; the runner's sampled means
are independently reproduced as a check.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
from run_unified_doc_spatial_v2 import read_source

from river_graph.experiments.unified_spatial_protocol import (
    support_query_cells,
)
from river_graph.models.kgml_local_transport import FIT_ROLES
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(support,cells,weight,bias,months):
    cells=torch.as_tensor(cells,dtype=torch.long)
    stations=cells//months
    dates=cells[:,None]%months-torch.arange(11,-1,-1)[None,:]
    valid=dates>=0
    index=dates.clamp_min(0)
    selected=support[stations[:,None],index]
    decay_input=torch.cat([torch.zeros((*selected.shape[:2],1),dtype=selected.dtype),selected],dim=-1)
    activation=torch.nn.functional.linear(decay_input,weight,bias)
    gamma=torch.exp(-torch.relu(activation))
    values=gamma[valid]
    return {'gamma':gamma,'valid':valid,'activation':activation,'values':values,
            'support_nonzero_fraction':float((selected[valid]!=0).float().mean())}


def main():
    torch.set_num_threads(2)
    rows=[];sources=[]
    for partition in (142,143,144):
        for seed in (42,43,44):
            run=ROOT/'runs'/f'split{partition}_seed{seed}'
            config=json.loads((run/'config.json').read_text())
            summary=json.loads((run/'unseen_neutral.json').read_text())
            saved=json.loads((run/'clock_diagnostics.json').read_text())['unseen_neutral']
            source=Path(config['source_run'])
            _,_,dataset,split,_=read_source(source)
            expert=UnifiedDOCReconstructor.load(source,dataset,split)
            with torch.inference_mode():
                x,_=expert.residual.input_view(split,FIT_ROLES)
                months=x.shape[0]
                val_ids=np.unique(split['val']//months)
                support=torch.stack([x[...,9],x[...,-2],x[...,-1]],dim=-1).permute(1,0,2)[val_ids]
                if torch.count_nonzero(x[:,val_ids,-7]):
                    raise ValueError('Validation receiving stations unexpectedly have seen DOC')
                _,query=support_query_cells(split,target_role='val',k=0,n_months=months)
                compact=np.sort(np.searchsorted(val_ids,query//months)*months+query%months)
                payload=torch.load(run/'unseen_neutral.pt',weights_only=True)
                weight,bias=payload['decay']['weight'],payload['decay']['bias']
                initial=payload['initial_decay']
                distance=float(sum((payload['decay'][key].double()-initial[key].double()).square().sum()
                                   for key in ('weight','bias')).sqrt())
                sampled=evaluate(support,saved['sample_cells'],weight,bias,months)
                if float(sampled['values'].numpy().mean())!=saved['mean_gamma']:
                    raise ValueError('Sampled neutral gamma does not replay exactly')
                block=evaluate(support,compact,weight,bias,months)
                gamma=block['values'];active=block['activation'][block['valid']]>0
                whole_identity=bool(torch.all(gamma==1))
                rows.append({'split_seed':partition,'seed':seed,'n_validation_queries':len(compact),
                    'n_valid_window_steps':int(block['valid'].sum()),'hidden_size':gamma.shape[1],
                    'gamma_min':float(gamma.min()),'gamma_mean':float(gamma.mean()),
                    'gamma_max':float(gamma.max()),'gamma_equal_one_fraction':float((gamma==1).double().mean()),
                    'units_identity_for_all_validation_steps':int((gamma==1).all(0).sum()),
                    'positive_relu_activation_fraction':float(active.double().mean()),
                    'entire_decay_identity_on_validation':whole_identity,
                    'support_input_nonzero_fraction':block['support_nonzero_fraction'],
                    'decay_parameter_distance_recomputed':distance,
                    'decay_parameter_distance_saved':summary['decay_parameter_distance'],
                    'weight_tensors_unchanged':all(torch.equal(payload['decay'][key],initial[key]) for key in initial),
                    'sampled_gamma_exact_replay':True})
            for path in (run/'config.json',run/'unseen_neutral.pt',run/'unseen_neutral.json',
                         run/'clock_diagnostics.json',source/'complete.json'):
                sources.append({'path':str(path),'sha256':sha(path)})
            del expert,x,dataset,payload
    out=ROOT/'analysis'
    pd.DataFrame(rows).to_csv(out/'source_validation_decay_capacity.csv',index=False)
    (out/'source_validation_decay_capacity_sources.json').write_text(json.dumps({
        'role':'source-validation-only diagnostic; no model selection or target labels',
        'script_sha256':sha(__file__),'sources':sources},indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()
