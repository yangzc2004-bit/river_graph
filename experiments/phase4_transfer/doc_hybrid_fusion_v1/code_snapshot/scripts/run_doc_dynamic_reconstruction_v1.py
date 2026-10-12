"""Four matched directional RiverDOCNet recipes and shared-input tree baselines.

Execute through the isolated run_ladder created by the companion launcher.
All source-validation choices are sealed before any new target scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from run_doc_fusion_benchmark_development_v1 import (
    bind,
    save_predictions,
    utc_now,
    verify,
)
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    role_labels,
    write_json,
)
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_domain_work import CANDIDATES as DOC_CANDIDATES
from river_graph.experiments.doc_domain_work import make_estimator as doc_estimator
from river_graph.experiments.doc_domain_work import optional_dependencies
from river_graph.experiments.doc_fusion_benchmark import CANDIDATES as TREE_CANDIDATES
from river_graph.experiments.doc_fusion_benchmark import (
    concentration_prediction,
    metrics,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_dynamic_reconstruction import (
    DynamicRiverDOC,
    change_metrics,
    dynamic_inputs,
    label_scales,
    reconstruction_loss,
)
from river_graph.models.river_architecture_comparison import covariate_inputs
from river_graph.models.river_relation_comparison import relation_graph

ROOT=Path("experiments/phase4_transfer/doc_dynamic_reconstruction_v1")
DATASET=Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES=Path("data/processed/graph_nodes_graphfix_st357.csv")
DAILY=Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1/daily_features.npz")
PARENT=Path("experiments/phase4_transfer/doc_river_relation_v2")
REGIONS=("1013","1019","0708","1030","1101")
SEEDS=(42,43,44)
NEURAL=("monthly_log","daily_log","monthly_joint","daily_joint")
ARMS=(*NEURAL,"rf_monthly","rf_daily","xgb_monthly","xgb_daily")
CODE=("scripts/run_ladder.py","scripts/run_doc_dynamic_reconstruction_v1.py",
      "scripts/start_doc_dynamic_reconstruction_v1.py","scripts/prepare_doc_dynamic_reconstruction_v1.py",
      "scripts/verify_doc_dynamic_reconstruction_v1.py","tests/test_doc_dynamic_reconstruction_v1.py",
      "src/river_graph/models/doc_dynamic_reconstruction.py",
      "src/river_graph/models/river_relation_comparison.py",
      "src/river_graph/models/river_architecture_comparison.py",
      "src/river_graph/experiments/doc_fusion_benchmark.py",
      "src/river_graph/experiments/doc_domain_work.py",
      "src/river_graph/experiments/provenance.py",
      "scripts/run_doc_fusion_benchmark_development_v1.py",
      "scripts/run_doc_river_architecture_comparison_v1.py")


def tasks():
    panels=[{"track":"space","region":region,"seed":seed,"mask":f"space_{region}","gap_months":0}
            for region in REGIONS for seed in SEEDS]
    panels.extend({"track":"time","region":"temporal","seed":seed,
                   "mask":f"time_g{gap}_s{seed}","gap_months":gap}
                  for gap in (3,6,12) for seed in SEEDS)
    return [{**p,"arm":arm,"id":f"{p['track']}_{p['region'] if p['track']=='space' else 'g'+str(p['gap_months'])}_{arm}_s{p['seed']}"}
            for p in panels for arm in ARMS]


def freeze(root,repository,workers):
    preparation=json.loads((root/"preparation.json").read_text())
    if preparation["dataset_sha256"]!=sha256_file(DATASET) or preparation["daily_sha256"]!=sha256_file(DAILY):
        raise ValueError("audited inputs changed")
    if preparation["raw_reproduced_doc_cells"]!=22571 or preparation["doc_monthly_mean_max_difference"]>1e-4:
        raise ValueError("raw DOC calendar-month means do not reproduce frozen labels")
    original=json.loads((PARENT/"protocol.json").read_text())
    masks={}
    for name in sorted({t["mask"] for t in tasks()}):
        src=PARENT/"masks"/f"{name}.npz";dst=root/"masks"/src.name
        dst.parent.mkdir(exist_ok=True)
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise ValueError("copied mask changed")
        if not dst.exists():dst.write_bytes(src.read_bytes())
        masks[name]=sha256_file(dst)
        if masks[name]!=original["mask_sha256"][name]:raise ValueError("parent mask changed")
    settings={**original["settings"]["space"],"torch_threads":1}
    protocol={"experiment":ROOT.name,"tasks":tasks(),"regions":list(REGIONS),"seeds":list(SEEDS),
              "arms":list(ARMS),"neural_arms":list(NEURAL),"settings":settings,"workers":workers,
              "dataset":str(DATASET),"dataset_sha256":sha256_file(DATASET),"nodes_sha256":sha256_file(NODES),
              "daily":str(DAILY),"daily_sha256":sha256_file(DAILY),"preparation_sha256":sha256_file(root/"preparation.json"),
              "mask_sha256":masks,"runtime_snapshot":{name:sha256_file(name) for name in CODE},
              "joint_primary":["mae","log1p_mae"],"prior_paper_endpoints_modified":False,
              "diagnostics":["station_centered_r2","log1p_station_centered_r2","within_station_sd_ratio","monthly_delta_mae","monthly_delta_correlation"],
              "neural_matching":"same 12853 parameter tensors, seeded initialization, MLP, 12-month GRU width12 and upstream/downstream/global heads; monthly controls zero eight added channels",
              "extra_inputs":"eight frozen bounded within-month daily-flow descriptors/support flags; same station/month footprint; no additional DOC, future-month driver or physical process states",
              "loss_joint":"standardized log1p MSE +0.25 source-SD-scaled native MSE +0.25 source-delta-RMS-scaled consecutive-month delta MSE; only pairs with both training labels",
              "batch_rule":"same shuffled current-month anchors for all arms; all arms also compute previous-month states, but base log loss uses current anchors only; constant expected-cell/pair denominators",
              "checkpoint_selection":"same native source-validation MAE, max60 epochs/min15/patience10; no target selection",
              "global_selection":"neural recipes only; validation relative native/log1p MAE to monthly_log; equal endpoint/task/space-time weights; select smallest score before target scoring; change diagnostics never replace primary endpoints",
              "tree_candidates":{"rf":TREE_CANDIDATES["random_forest"],"xgb":DOC_CANDIDATES["doc_xgboost"]},
              "tree_selection":"validation native MAE then log-MAE; same local environmental/history input as neural arms; no station identity or DOC history",
              "scope":"previously inspected Mississippi development tasks, not independent confirmation; observed-query month means, not naturally missing truth or daily events",
              "aggregation":"space: pool five held regions per seed then average3 seed metrics; time: average3 mask/optimization-seed metrics per gap; no prediction ensemble",
              "original_dispatcher_sha256":sha256_file(repository/"scripts/run_ladder.py"),
              "versions":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__}}
    if (root/"protocol.json").exists():
        if json.loads((root/"protocol.json").read_text())!=protocol:raise ValueError("frozen protocol changed; create new version")
    else:
        write_json(root/"protocol.json",protocol)
        for name in CODE:
            dst=root/"code_snapshot"/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json",{"protocol_hash":digest(protocol),"frozen_at":utc_now(),
                   "code_commit":subprocess.check_output(["git","-C",str(repository),"rev-parse","HEAD"],text=True).strip(),
                   "workspace_dirty":bool(subprocess.check_output(["git","-C",str(repository),"status","--porcelain"],text=True)),
                   "generator_script_sha256":sha256_file(CODE[1]),"runtime_snapshot_hash":digest(protocol["runtime_snapshot"]),
                   "dataset_sha256":protocol["dataset_sha256"],"nodes_sha256":protocol["nodes_sha256"],"mask_sha256":masks})
    return protocol


def config(protocol,task):
    return {"experiment":ROOT.name,**task,"protocol_hash":digest(protocol),"dataset_sha256":protocol["dataset_sha256"],
            "mask_sha256":protocol["mask_sha256"][task["mask"]],"daily_sha256":protocol["daily_sha256"],
            "runtime_snapshot_hash":digest(protocol["runtime_snapshot"])}


def context(root,task,data,daily):
    with np.load(root/"masks"/f"{task['mask']}.npz",allow_pickle=False) as a:
        split={r:a[r].copy() for r in ("train","val","test")}
    months=data["y"].shape[1];rows={r:np.unique(c//months) for r,c in split.items()}
    for a,b in (("train","val"),("train","test"),("val","test")):
        assert not np.intersect1d(split[a],split[b]).size
        if task["track"]=="space":assert not np.intersect1d(rows[a],rows[b]).size
    view=task["arm"].split("_")[0] if task["arm"] in NEURAL else task["arm"].split("_")[1]
    original=covariate_inputs({k:v for k,v in data.items() if k not in ("y","y_mask")},rows["train"])
    inputs=dynamic_inputs(original,daily,view)
    graphs={r:relation_graph(data["edge_index"],inputs["order"],rows["train"] if r=="train" else np.union1d(rows["train"],rows[r]),"direction_heads")
            for r in ("train","val","test")}
    return {"split":split,"rows":rows,"inputs":inputs,"graphs":graphs,"months":months,"view":view}


def make_frame(data,task,cells,pred):
    return pd.DataFrame({"cell":cells,"station":cells//data["y"].shape[1],"month":cells%data["y"].shape[1],
                         "y_true":np.asarray(data["y"]).ravel()[cells],"y_pred":pred,"high_doc":False,
                         "target_huc4":task["region"],"model_name":task["arm"],"seed":task["seed"],
                         "track":task["track"],"gap_months":task["gap_months"]})


def neural_cells(model,ctx,cells,scales,batch):
    role="val" if np.array_equal(cells,ctx["split"]["val"]) else "test"
    graph=ctx["graphs"][role];times=np.unique(cells%ctx["months"])
    norm={"mean":scales["log_mean"],"std":scales["log_std"]}
    grid=native_prediction(predict(model,ctx["inputs"],graph,times,batch),norm)
    return grid[np.searchsorted(times,cells%ctx["months"]),np.searchsorted(graph["rows"].numpy(),cells//ctx["months"])]


def tree_features(ctx,cells):
    node,month=cells//ctx["months"],cells%ctx["months"]
    windows=ctx["inputs"]["windows"].numpy()[month,node]
    if ctx["view"]=="monthly":windows=windows[...,:8]
    return np.column_stack([ctx["inputs"]["environment"].numpy()[node],windows.reshape(len(cells),-1),ctx["inputs"]["season"].numpy()[month]])


def train_neural(ctx,task,data,settings,progress):
    torch.manual_seed(task["seed"])
    model=DynamicRiverDOC(**{k:settings[k] for k in ("hidden","temporal_hidden","heads","layers","dropout")})
    train_graph=ctx["graphs"]["train"];val_graph=ctx["graphs"]["val"]
    train_y=role_labels(data,ctx["split"],"train",train_graph["rows"].numpy())
    val_y=role_labels(data,ctx["split"],"val",val_graph["rows"].numpy())
    scales=label_scales(train_y)
    active=torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    batches=int(np.ceil(len(active)/settings["batch_months"]))
    denominators={"cells":scales["training_cells"]/batches,"pairs":max(scales["training_adjacent_pairs"]/batches,1.)}
    previous_y=torch.cat([torch.full_like(train_y[:1],float("nan")),train_y[:-1]])
    val_months=torch.where(torch.isfinite(val_y).any(1))[0].numpy();val_truth=val_y[val_months].numpy();valid=np.isfinite(val_truth)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    rng=np.random.default_rng(task["seed"]);rows=train_graph["rows"]
    loss_name=task["arm"].split("_")[1];best=float("inf");stale=0;state=None;trace=[];started=time.monotonic()
    for epoch in range(1,settings["max_epochs"]+1):
        model.train();permutation=rng.permutation(active);total=0.
        for start in range(0,len(permutation),settings["batch_months"]):
            current=permutation[start:start+settings["batch_months"]]
            previous=np.maximum(current-1,0);union=np.union1d(current,previous)
            optimizer.zero_grad(set_to_none=True)
            all_z=model(ctx["inputs"]["windows"][union][:,rows],ctx["inputs"]["environment"][rows],ctx["inputs"]["season"][union],train_graph)
            z=all_z[np.searchsorted(union,current)];previous_z=all_z[np.searchsorted(union,previous)]
            value=reconstruction_loss(z,train_y[current],previous_z,previous_y[current],scales,denominators,loss_name)
            if not torch.isfinite(value):raise ValueError("nonfinite objective")
            value.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient):raise ValueError("nonfinite gradient")
            optimizer.step();total+=float(value.detach())
        pred=native_prediction(predict(model,ctx["inputs"],val_graph,val_months,settings["batch_months"]),
                               {"mean":scales["log_mean"],"std":scales["log_std"]})
        score=float(np.abs(pred[valid]-val_truth[valid]).mean())
        if score<best-1e-6:
            best=score;best_epoch=epoch;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        row={"epoch":epoch,"objective_sum":total,"validation_mae":score,"best_epoch":best_epoch,"elapsed_seconds":time.monotonic()-started};trace.append(row)
        if epoch==1 or epoch%5==0:progress(row)
        if epoch>=settings["min_epochs"] and stale>=settings["patience"]:break
    model.load_state_dict(state)
    return model,{"best_epoch":best_epoch,"epochs_run":epoch,"validation_mae":best,
                  "training_seconds":time.monotonic()-started,"parameters":sum(p.numel() for p in model.parameters()),"scales":scales},trace


def fit_task(root,task,protocol,data,daily):
    directory=root/"fits"/task["id"];identity=config(protocol,task)
    if (directory/"selection_complete.json").exists():verify(directory,"selection_complete.json",identity);return
    directory.mkdir(parents=True,exist_ok=True);write_json(directory/"config.json",identity)
    ctx=context(root,task,data,daily);names=["config.json","normalization.json","info.json","selection.json","validation_predictions.parquet","validation_predictions.meta.json"]
    settings=protocol["settings"]
    if task["arm"] in NEURAL:
        progress=lambda row:write_json(root/f"progress_{task['id']}.json",{"stage":"training",**task,**row})
        model,info,trace=train_neural(ctx,task,data,settings,progress)
        torch.save(model.state_dict(),directory/"model.pt");write_json(directory/"trace.json",trace)
        names.extend(["model.pt","trace.json"])
        cells=ctx["split"]["val"];prediction=neural_cells(model,ctx,cells,info["scales"],settings["batch_months"])
        scores=[]
    else:
        family=task["arm"].split("_")[0];train=ctx["split"]["train"];cells=ctx["split"]["val"]
        x=tree_features(ctx,train);v=tree_features(ctx,cells);y=np.asarray(data["y"]).ravel()[train]
        scores=[];chosen=None;key=None;started=time.monotonic()
        recipes=protocol["tree_candidates"][family]
        for recipe_name,recipe in recipes.items():
            target=recipe["target"]
            if family=="rf":
                candidate=RandomForestRegressor(n_estimators=300,max_features=1.,n_jobs=1,random_state=task["seed"],**{k:v for k,v in recipe.items() if k!="target"})
            else:
                candidate=doc_estimator("doc_xgboost",recipe,task["seed"],1)
            candidate.fit(x,np.log1p(y) if target=="log1p" else y)
            p=concentration_prediction(candidate.predict(v),target)[0];part=make_frame(data,task,cells,p);m=metrics(part)
            score=(m["mae"],m["log1p_mae"],recipe_name);scores.append({"recipe":recipe_name,"target":target,**m})
            if key is None or score<key:key=score;chosen=candidate;prediction=p;transform=target;selected_recipe=recipe_name
        model=chosen;info={"target":transform,"selected_recipe":selected_recipe,"training_seconds":time.monotonic()-started,"features":x.shape[1],"selected_validation_mae":key[0]}
        joblib.dump(model,directory/"model.joblib",compress=3);names.append("model.joblib")
    part=make_frame(data,task,cells,prediction)
    write_json(directory/"normalization.json",ctx["inputs"]["normalization"])
    write_json(directory/"info.json",info)
    write_json(directory/"selection.json",{"test_used":False,"validation_metrics":{**metrics(part),**change_metrics(part)},"candidate_scores":scores})
    save_predictions(directory,"validation_predictions",part,identity);bind(directory,"selection_complete.json",identity,names)
    write_json(root/f"progress_{task['id']}.json",{"stage":"selection_complete",**task})
    print(json.dumps({"stage":"selection_complete","task":task["id"],"seconds":info["training_seconds"]}),flush=True)


def choose(root,protocol):
    rows=[{**t,**json.loads((root/"fits"/t["id"]/"selection.json").read_text())["validation_metrics"]} for t in protocol["tasks"] if t["arm"] in NEURAL]
    frame=pd.DataFrame(rows);base=frame[frame.arm=="monthly_log"][["track","mask","seed","mae","log1p_mae"]]
    pair=frame.merge(base,on=["track","mask","seed"],suffixes=("","_base"),validate="many_to_one")
    pair["joint_ratio"]=.5*(pair.mae/pair.mae_base+pair.log1p_mae/pair.log1p_mae_base)
    by_track=pair.groupby(["arm","track"]).joint_ratio.mean().unstack();score=by_track.mean(axis=1)
    result={"test_used":False,"selected_arm":min(NEURAL,key=lambda a:score.loc[a]),"validation_joint_ratios":score.to_dict(),"validation_track_ratios":by_track.to_dict(orient="index")}
    write_json(root/"selected_recipe.json",result);return result


def score_task(root,task,protocol,data,daily):
    directory=root/"fits"/task["id"];identity=config(protocol,task)
    if (directory/"test_complete.json").exists():verify(directory,"test_complete.json",identity);return
    verify(directory,"selection_complete.json",identity)
    ctx=context(root,task,data,daily);cells=ctx["split"]["test"];info=json.loads((directory/"info.json").read_text())
    if task["arm"] in NEURAL:
        model=DynamicRiverDOC(**{k:protocol["settings"][k] for k in ("hidden","temporal_hidden","heads","layers","dropout")})
        model.load_state_dict(torch.load(directory/"model.pt",weights_only=True,map_location="cpu"))
        p=neural_cells(model,ctx,cells,info["scales"],protocol["settings"]["batch_months"])
    else:
        model=joblib.load(directory/"model.joblib");p=concentration_prediction(model.predict(tree_features(ctx,cells)),info["target"])[0]
    save_predictions(directory,"predictions",make_frame(data,task,cells,p),identity)
    bind(directory,"test_complete.json",identity,["predictions.parquet","predictions.meta.json"])


def analyze(root,protocol):
    all_predictions=pd.concat([pd.read_parquet(root/"fits"/t["id"]/"predictions.parquet") for t in protocol["tasks"]],ignore_index=True)
    rows=[{"track":track,"gap_months":int(gap),"seed":int(seed),"model_name":arm,**metrics(part),**change_metrics(part)}
          for (track,gap,seed,arm),part in all_predictions.groupby(["track","gap_months","seed","model_name"])]
    by_seed=pd.DataFrame(rows);summary=by_seed.groupby(["track","gap_months","model_name"],as_index=False).mean(numeric_only=True)
    for i,row in summary.iterrows():
        base=summary[(summary.track==row.track)&(summary.gap_months==row.gap_months)&(summary.model_name=="monthly_log")].iloc[0]
        summary.loc[i,"mae_improvement_percent"]=100*(1-row.mae/base.mae)
        summary.loc[i,"log1p_mae_improvement_percent"]=100*(1-row.log1p_mae/base.log1p_mae)
        summary.loc[i,"delta_mae_improvement_percent"]=100*(1-row.monthly_delta_mae/base.monthly_delta_mae)
    regions=[{"region":region,"seed":int(seed),"model_name":arm,**metrics(part),**change_metrics(part)}
             for (region,seed,arm),part in all_predictions[all_predictions.track=="space"].groupby(["target_huc4","seed","model_name"])]
    out=root/"analysis";out.mkdir(exist_ok=True)
    by_seed.to_csv(out/"by_seed.csv",index=False);summary.to_csv(out/"summary.csv",index=False)
    pd.DataFrame(regions).to_csv(out/"by_region_seed.csv",index=False)
    bind(out,"complete.json",{"protocol_hash":digest(protocol)},["by_seed.csv","summary.csv","by_region_seed.csv"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--repository",type=Path,default=Path.cwd())
    parser.add_argument("--workers",type=int,default=4);parser.add_argument("--freeze-only",action="store_true");parser.add_argument("--task")
    args=parser.parse_args();protocol=json.loads((args.root/"protocol.json").read_text()) if args.task else freeze(args.root,args.repository,args.workers)
    if args.freeze_only:print(json.dumps({"stage":"frozen","tasks":len(protocol["tasks"])}));return
    optional_dependencies();torch.set_num_threads(1)
    data=torch.load(DATASET,weights_only=False,map_location="cpu")
    with np.load(DAILY,allow_pickle=False) as a:daily=a["full"].copy()
    if args.task:
        task=next(t for t in protocol["tasks"] if t["id"]==args.task)
        with threadpool_limits(limits=1):fit_task(args.root,task,protocol,data,daily)
        return
    pending=list(protocol["tasks"]);running=[];logs=args.root/"worker_logs";logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running)<args.workers:
            task=pending.pop(0);log=(logs/f"{task['id']}.log").open("a")
            process=subprocess.Popen([os.sys.executable,"scripts/run_ladder.py","--root",str(args.root),"--task",task["id"]],stdout=log,stderr=subprocess.STDOUT)
            running.append((task,process,log))
        time.sleep(.5);remaining=[]
        for task,process,log in running:
            status=process.poll()
            if status is None:remaining.append((task,process,log))
            else:
                log.close()
                if status:raise RuntimeError(f"worker failed: {task['id']}; inspect its log")
        running=remaining
    choose(args.root,protocol)
    write_json(args.root/"all_selections.json",{"protocol_hash":digest(protocol),"fit_receipts":{t["id"]:sha256_file(args.root/"fits"/t["id"]/"selection_complete.json") for t in protocol["tasks"]},
                                               "selected_recipe_sha256":sha256_file(args.root/"selected_recipe.json"),"test_used":False,"frozen_at":utc_now()})
    with threadpool_limits(limits=1):
        for task in protocol["tasks"]:score_task(args.root,task,protocol,data,daily)
    analyze(args.root,protocol)
    write_json(args.root/"batch_complete.json",{"protocol_hash":digest(protocol),"tasks":len(protocol["tasks"]),"completed_at":utc_now()})
    print(json.dumps({"stage":"complete","tasks":len(protocol["tasks"])}),flush=True)


if __name__=="__main__":main()
