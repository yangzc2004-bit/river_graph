"""Independent original-label, metric, selection and checkpoint replay checks."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_dynamic_reconstruction_v1 import (
    CODE,
    DATASET,
    ROOT,
    context,
    neural_cells,
    tree_features,
)
from run_doc_fusion_benchmark_development_v1 import verify
from run_doc_river_architecture_comparison_v1 import digest, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_domain_work import optional_dependencies
from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_dynamic_reconstruction import DynamicRiverDOC


def independent(part):
    y,p=part.y_true.to_numpy(float),part.y_pred.to_numpy(float)
    residual=p-y;sse=float(np.dot(residual,residual));sst=float(np.sum((y-y.mean())**2))
    station_mean=part.groupby("station")[["y_true","y_pred"]].transform("mean")
    a=y-station_mean.y_true.to_numpy(float);b=p-station_mean.y_pred.to_numpy(float)
    return {"mae":float(sum(abs(residual))/len(y)),"log1p_mae":float(np.mean(abs(np.log1p(p)-np.log1p(y)))),
            "rmse":float((sse/len(y))**.5),"r2":1-sse/sst,"station_centered_r2":1-float(np.sum((b-a)**2))/float(np.sum(a*a))}


def main():
    optional_dependencies();torch.set_num_threads(1)
    protocol=json.loads((ROOT/"protocol.json").read_text());gate=json.loads((ROOT/"all_selections.json").read_text())
    assert gate["protocol_hash"]==digest(protocol) and gate["selected_recipe_sha256"]==sha256_file(ROOT/"selected_recipe.json")
    assert protocol["joint_primary"]==["mae","log1p_mae"] and not protocol["prior_paper_endpoints_modified"]
    assert sha256_file(DATASET)==protocol["dataset_sha256"]
    for name in CODE:
        if name=="scripts/run_ladder.py":continue
        assert sha256_file(name)==protocol["runtime_snapshot"][name],name
    data=torch.load(DATASET,weights_only=False,map_location="cpu")
    with np.load(protocol["daily"],allow_pickle=False) as a:daily=a["full"].copy()
    parts=[];validation=[];replay_error=0.;seen={}
    with threadpool_limits(limits=1):
        for task in protocol["tasks"]:
            directory=ROOT/"fits"/task["id"]
            assert gate["fit_receipts"][task["id"]]==sha256_file(directory/"selection_complete.json")
            verify(directory,"selection_complete.json");verify(directory,"test_complete.json")
            ctx=context(ROOT,task,data,daily);info=json.loads((directory/"info.json").read_text())
            if task["arm"] in protocol["neural_arms"]:
                model=DynamicRiverDOC(**{k:protocol["settings"][k] for k in ("hidden","temporal_hidden","heads","layers","dropout")})
                model.load_state_dict(torch.load(directory/"model.pt",weights_only=True,map_location="cpu"))
            else:model=joblib.load(directory/"model.joblib")
            for role,name in (("val","validation_predictions"),("test","predictions")):
                f=pd.read_parquet(directory/f"{name}.parquet").sort_values("cell");cells=f.cell.to_numpy()
                np.testing.assert_array_equal(cells,np.sort(ctx["split"][role]));assert not f.cell.duplicated().any()
                np.testing.assert_array_equal(f.y_true.to_numpy(),np.asarray(data["y"]).ravel()[cells])
                assert np.isfinite(f.y_pred).all() and (f.y_pred>=0).all()
                if task["arm"] in protocol["neural_arms"]:
                    prediction=neural_cells(model,ctx,ctx["split"][role],info["scales"],protocol["settings"]["batch_months"])
                    prediction=pd.Series(prediction,index=ctx["split"][role]).loc[cells].to_numpy()
                else:
                    raw=model.predict(tree_features(ctx,cells)).astype(float)
                    prediction=np.clip(np.expm1(np.minimum(raw,15.)) if info["target"]=="log1p" else raw,0.,np.expm1(15.))
                error=float(np.max(abs(prediction-f.y_pred.to_numpy())));replay_error=max(replay_error,error)
                assert error<1e-5,(task["id"],role,error)
                if role=="test":parts.append(f)
                else:validation.append({**task,**independent(f)})
                identity=(task["mask"],task["seed"],role)
                fingerprint=f[["cell","y_true"]].to_numpy()
                if identity in seen:np.testing.assert_array_equal(fingerprint,seen[identity])
                else:seen[identity]=fingerprint
    all_predictions=pd.concat(parts,ignore_index=True);rows=[]
    for (track,gap,seed,arm),f in all_predictions.groupby(["track","gap_months","seed","model_name"]):
        rows.append({"track":track,"gap_months":gap,"seed":seed,"model_name":arm,**independent(f)})
    expected=pd.DataFrame(rows).groupby(["track","gap_months","model_name"]).mean(numeric_only=True)
    actual=pd.read_csv(ROOT/"analysis/summary.csv").set_index(["track","gap_months","model_name"])
    diffs=[]
    for k,r in expected.iterrows():
        for metric in ("mae","log1p_mae","rmse","r2","station_centered_r2"):
            error=abs(float(r[metric])-float(actual.loc[k,metric]));assert error<1e-6,(k,metric,error);diffs.append(error)
    v=pd.DataFrame(validation);v=v[v.arm.isin(protocol["neural_arms"])]
    base=v[v.arm=="monthly_log"][["mask","seed","mae","log1p_mae"]]
    paired=v.merge(base,on=["mask","seed"],suffixes=("","_base"),validate="many_to_one")
    paired["joint"]=(paired.mae/paired.mae_base+paired.log1p_mae/paired.log1p_mae_base)/2
    ratios=paired.groupby(["arm","track"]).joint.mean().groupby("arm").mean()
    choice=json.loads((ROOT/"selected_recipe.json").read_text())
    assert choice["selected_arm"]==min(protocol["neural_arms"],key=lambda a:ratios.loc[a])
    report={"passed":True,"tasks":len(protocol["tasks"]),"replayed_query_roles":2*len(protocol["tasks"]),
            "prediction_replay_max_abs":replay_error,"metric_recompute_max_abs":max(diffs),
            "matched_queries_and_original_truth":True,"validation_only_selection_recomputed":True,
            "unchanged_primary_endpoints":True,"analysis_sha256":sha256_file(ROOT/"analysis/summary.csv")}
    write_json(ROOT/"analysis/independent_audit.json",report);print(json.dumps(report))


if __name__=="__main__":main()
