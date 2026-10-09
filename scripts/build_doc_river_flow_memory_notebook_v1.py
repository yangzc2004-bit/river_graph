"""Build and execute the portable companion for the flow-memory evidence."""
from __future__ import annotations

import base64
import html
import json
import sys
import tempfile
from pathlib import Path

import nbformat
import pandas as pd
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path("experiments/phase4_transfer/doc_river_flow_memory_v1")


def main():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        md("# River structure and monthly DOC–flow memory\n\n"
           "Executed research companion, 2026-10-09. Source-role observations only. "
           "The main population has 206 stations, 49 HUC4 regions and 13,846 monthly pairs. "
           "Earlier-to-later prediction is possible at 72 stations. In the recent-period "
           "sensitivity, previous flow reduces log1p MAE by 9.80% relative to current flow, "
           "but the full-period improvement is not established. Structural blocks do not "
           "reliably predict the hydrologic-response differences across regions."),
        md("## Context and methods\n\n"
           "This follow-up adds preceding-calendar-month flow to the previously measured "
           "contemporary response. DOC is read only from the frozen source-training union "
           "142/143/144. River units, rather than months, govern inference. Temporal models "
           "fit season, trend and flow using earlier observations only. Geographic response "
           "models use five whole-HUC4 folds and Ridge alpha=10. Interval estimates preserve "
           "whole HUC4 groups and are conditional on saved predictions. All comparisons "
           "and sensitivities are retained; intervals are pointwise.\n\n"
           "### Key assumptions\n\n"
           "Flow is the monthly mean of NWIS daily flow, not instantaneous flow at DOC "
           "sampling. Previous-month association is hydrologic memory; it does not measure "
           "one-month transport. Seasonal/trend adjustment does not identify causal effects. "
           "Original NEON discharge cannot be acquired through the official endpoint "
           "without user-authorized authentication; the independent-flow replication is pending."),
        code("from pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n"
             "from IPython.display import Image, display\n\n"
             "repo = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())\n"
             "study = repo/'experiments/phase4_transfer/doc_river_flow_memory_v1'\n"
             "counts = pd.read_csv(study/'analysis/population_counts.csv')\n"
             "display(counts)"),
        md("## Paired later-period prediction\n\n"
           "Every arm predicts identical later station-months. The following arithmetic "
           "recomputes all 18 point estimates directly from saved predictions."),
        code("predictions = pd.read_parquet(study/'analysis/chronological_predictions.parquet')\n"
             "gains = pd.read_csv(study/'analysis/chronological_gains.csv')\n"
             "for r in gains.itertuples():\n"
             "    s = predictions[predictions.population.eq(r.population)].copy()\n"
             "    s['error'] = abs(s.doc-s.prediction) if r.space=='native' else abs(np.log1p(s.doc)-s.log_pred)\n"
             "    station_errors = s.groupby(['station','arm']).error.mean().unstack()\n"
             "    base, candidate = station_errors[r.reference].mean(), station_errors[r.candidate].mean()\n"
             "    np.testing.assert_allclose(100*(base-candidate)/base, r.gain_pct, atol=1e-10)\n"
             "display(gains[gains.candidate.eq('flow_memory') & gains.reference.eq('current_flow')]"
             "[['population','space','gain_pct','ci_low_pct','ci_high_pct','positive_stations','n_stations']].round(3))\n"
             "display(Image(filename=str(study/'figures/river_flow_memory_evidence_cn.png')))"),
        md("## Geographic structural information\n\n"
           "The response being predicted is measured log1p DOC sensitivity to current "
           "or previous log flow, rather than production-model DOC. Static shape classes "
           "are not reselected using outcomes."),
        code("structure = pd.read_csv(study/'analysis/descriptor_gains.csv')\n"
             "display(structure[structure.candidate.eq('branching') & structure.response.eq('previous_response')]"
             "[['population','gain_pct','ci_low_pct','ci_high_pct','n_stations','n_huc4']].round(3))\n"
             "ledger = pd.read_csv(study/'analysis/chronological_eligibility.csv')\n"
             "s = ledger[ledger.population.eq('all_source_months') & ledger.status.eq('included') & ledger.late_status.eq('included')]\n"
             "display(pd.DataFrame([{'response':r,'stations':len(s),'same_sign_fraction':"
             "np.mean((s[f'train_{r}_response']>0)==(s[f'late_{r}_response']>0))} for r in ['current','previous']]))\n"
             "display(Image(filename=str(study/'figures/river_flow_response_stability_cn.png')))"),
        md("## Model destination\n\n"
           "The evidence supports checking conditional use of hydrologic history. It does "
           "not provide validated shape-specific lag weights or physical concentration "
           "constraints. The next model comparison should keep the current complete DOC "
           "predictor as the base, control identical static structure inputs, and contrast "
           "true upstream state histories against same-input no-message and source-matched "
           "non-upstream controls. Previous graph readout experiments already found no "
           "increment to the full model; a new test must change the dynamic operator, "
           "not relabel the old readout or claim this diagnostic's 9.8% as a GNN gain.\n\n"
           "Inputs and calculations are specified in study_plan.md, analysis_sources.json, "
           "verification.json and scripts/analyze_doc_river_flow_memory_v1.py."),
    ])
    notebook.metadata["kernelspec"] = {"display_name": "Python (river flow study)", "language": "python", "name": "river-flow-study"}
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="river-flow-notebook-") as temporary:
        kernel = Path(temporary)/"river-flow-study"
        kernel.mkdir()
        (kernel/"kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                                      "display_name": "river flow study", "language": "python"}))
        manager = KernelManager(kernel_name="river-flow-study", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        NotebookClient(notebook, km=manager, timeout=120, resources={"metadata": {"path": str(Path.cwd())}}).execute()
    nbformat.validate(notebook)
    path = ROOT/"river_flow_memory_research.ipynb"
    nbformat.write(notebook, path)
    # An offline QA preview uses the exact executed cell outputs, not a second
    # calculation or an alternative report. It stays outside committed results.
    fragments = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            fragments.append("<div class='prose'>"+html.escape(cell.source).replace("\n", "<br>")+"</div>")
        else:
            fragments.append("<details><summary>Executed Python</summary><pre>"+html.escape(cell.source)+"</pre></details>")
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    fragments.append("<pre>"+html.escape(output.text)+"</pre>")
                elif "image/png" in output.get("data", {}):
                    encoded = output.data["image/png"]
                    base64.b64decode(encoded, validate=True)
                    fragments.append(f"<img alt='Executed scientific figure' src='data:image/png;base64,{encoded}'>")
                elif "text/html" in output.get("data", {}):
                    fragments.append(output.data["text/html"])
                elif "text/plain" in output.get("data", {}):
                    fragments.append("<pre>"+html.escape(output.data["text/plain"])+"</pre>")
    preview = Path(tempfile.gettempdir())/"river-flow-memory-notebook-preview.html"
    preview.write_text("<!doctype html><meta charset='utf-8'><style>body{font:16px/1.6 Arial,sans-serif;max-width:1150px;margin:35px auto;color:#253F49} .prose{margin:30px 0} details{margin:15px 0} pre{overflow:auto;background:#f5f7f7;padding:12px;font-size:12px} table{border-collapse:collapse;font-size:12px;max-width:100%;margin:20px 0} td,th{padding:7px;border-bottom:1px solid #dde4e4} img{display:block;width:100%;margin:25px 0}</style>"+"\n".join(fragments))
    temporal = pd.read_csv(ROOT/"analysis"/"chronological_gains.csv")
    temporal = temporal[temporal.candidate.eq("flow_memory") & temporal.reference.eq("current_flow")]
    structure = pd.read_csv(ROOT/"analysis"/"descriptor_gains.csv")
    def columns(names):
        return [{"field": name, "label": label} for name, label in names]
    fields = [("population", "观测范围"), ("gain_pct", "MAE降幅（%）"),
              ("ci_low_pct", "95%下限（%）"), ("ci_high_pct", "95%上限（%）"), ("n_stations", "站点数")]
    temporal_fields = [("population", "观测范围"), ("space", "评价尺度"), *fields[1:]]
    structure_fields = [("population", "观测范围"), ("response", "响应描述"), ("candidate", "结构信息"), *fields[1:]]
    payload = {"schemaVersion": 1, "items": [
        {"id": "monthly-doc-flow-history", "title": "前月流量的信息价值依赖观测范围", "queries": [{
            "id": "chronological-previous-flow", "source": {
                "label": "ST357源角色DOC与实测月流量", "files": [{"label": "chronological_gains.csv"}, {"label": "verification.json"}],
                "metricDefinitions": [{"label": "历史流量增益", "definition": "同站前期拟合、后期评价；相对同月流量模型，加入前一个日历月流量后的MAE降幅。先在站内平均月份，再站点等权。"}],
                "filters": ["DOC仅来自142/143/144源训练角色的观测单元", "当月及前月流量均为实测正值", "前期至少24个月，后期至少12个月"],
                "caveats": ["这是简单水文响应诊断，9.8%不是当前神经DOC模型的提升。", "2009年以来为敏感性样本；全时段和温度调整样本未稳定复现该增益。", "95%区间来自5000次整片HUC4配对抽样，条件于已拟合预测，未计入重新拟合与多重比较调整。", "月流量不等于DOC采样时瞬时流量，前月关联不代表一个月的物理传输时间。"],
                "evidenceFlow": [{"kind": "calculation", "title": "相同后期月份上的增量", "detail": "100×(当月流量模型MAE−当月及前月模型MAE)/当月流量模型MAE。"},
                                 {"kind": "validation", "title": "独立复算", "detail": "全部18项时间比较与36项结构比较的点估计已从保存预测独立复算；各时间臂query一致，扰动后期DOC不改变预测。"}]},
            "columns": columns(temporal_fields), "rows": temporal[[n for n, _ in temporal_fields]].to_dict("records"),
            "reportingPeriod": "1972–2026源角色月观测；近期敏感性为2009年以来"}]},
        {"id": "form-response-transfer", "title": "静态河网结构尚不能稳定预测水文响应差异", "queries": [{
            "id": "geographic-response-blocks", "source": {
                "label": "真实河网形态与DOC水文响应", "files": [{"label": "descriptor_gains.csv"}, {"label": "structure_response_modifiers.csv"}],
                "metricDefinitions": [{"label": "结构响应信息", "definition": "五个HUC4整区留出折中，固定Ridge alpha=10；在相同环境与面积背景下加入外轮廓、支流、路径或全部结构，预测站点实测DOC—流量响应描述。"}],
                "caveats": ["响应描述预测误差不等于完整模型DOC重建误差。", "结构系数的关联分析不识别因果效应；结构对典型DOC水平的信息价值与动态响应是不同问题。"],
                "evidenceFlow": [{"kind": "method", "title": "地理留出", "detail": "整片HUC4不进入该折训练，标准化与缺失填充在训练折拟合。误差在折内站点平均、折间等权。"}]},
            "columns": columns(structure_fields), "rows": structure[[n for n, _ in structure_fields]].to_dict("records")}]},
        {"id": "independent-flow-access", "title": "独立NEON水文响应复验尚待授权数据下载", "queries": [{
            "id": "neon-discharge-access", "source": {"label": "NEON连续流量官方数据接口",
                "links": [{"label": "官方流量产品说明", "url": "https://www.neonscience.org/resources/learning-hub/tutorials/continuous-discharge-intro"},
                          {"label": "官方接口授权要求", "url": "https://data.neonscience.org/data-api/endpoints/data/"}],
                "caveats": ["对官方BLUE/2024-06流量端点的一次无凭证请求返回403，未获取原始流量；独立DOC—流量复验尚未执行。"],
                "evidenceFlow": [{"kind": "source", "title": "已确认数据来源", "detail": "DP4.00130.001是官方连续流量产品，下载端点要求已授权账户API token。"}]}}]},
    ]}
    (ROOT/"inline_sources.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    print(f"Executed and validated: {path}")
    print(f"Offline preview: {preview}")


if __name__ == "__main__":
    main()
