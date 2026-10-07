"""Actual network maps and DOC evidence for geometry-only internal profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import NullFormatter
from plot_doc_river_planform_v1 import draw_network

from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import read_cached_lines

ROOT = Path("experiments/phase4_transfer/doc_river_internal_structure_v1")
CACHE = Path("data/raw/river_planform_v1")
PROFILE_COLORS = {1: "#9A815E", 2: "#BE826D", 3: "#257F88", 4: "#66749F"}
OUTLINE_COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def geometry_digest(connection, comids):
    lines = read_cached_lines(connection, comids)
    if set(lines) != set(map(int, comids)):
        raise ValueError("every representative reach must have cached geometry")
    h = hashlib.sha256()
    for cid in sorted(lines):
        h.update(str(cid).encode()+b"\0"+lines[cid].wkb)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 220, "pdf.fonttype": 42})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    sources = [a/f"{name}.csv" for name in ("network_profiles", "representatives", "outline_profile_counts",
        "normalized_routing_summary", "observed_buffer_summary")]+[a/"representative_pulses.parquet", ROOT/"config.json"]
    networks, reps, counts, routing, observed = [pd.read_csv(p, dtype={"station": str, "huc_cd": str}) for p in sources[:5]]
    traces = pd.read_parquet(a/"representative_pulses.parquet")
    config = json.loads((ROOT/"config.json").read_text())
    names = {0: "无汇合点", 1: "汇合较不均衡\n路径较集中", 2: "汇合较不均衡\n路径较分散",
        3: "汇合较均衡\n路径较集中", 4: "汇合较均衡\n路径较分散"} if cn else {
        0: "No junctions", 1: "Less balanced\nConcentrated paths", 2: "Less balanced\nDispersed paths",
        3: "More balanced\nConcentrated paths", 4: "More balanced\nDispersed paths"}
    shapes = ("细长、多支流型", "主干主导、稀支流型", "宽展、多支流型") if cn else (
        "Elongated / tributary-rich", "Mainstem / sparse", "Broad / tributary-rich")
    suffix, written = ("_cn" if cn else ""), []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, facecolor="white", bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax = axes[0]
    for k in (1, 2, 3):
        f = networks[networks.cluster.eq(k)]
        ax.scatter(f.tributary_balance, f.path_cv, color=OUTLINE_COLORS[k], alpha=.65, s=18, label=shapes[k-1])
    ax.axvline(config["balance_cut"], linestyle="--", color="#76868B", linewidth=1.)
    ax.axhline(config["path_cut"], linestyle="--", color="#76868B", linewidth=1.)
    ax.set(xlim=(-.01, .45), ylim=(.275, .65), xlabel="汇合均衡度（典型小支流面积份额代理）" if cn else "Junction balance (median non-dominant area share)",
        ylabel="路径长短差异（面积加权 CV）" if cn else "Path dispersion (area-weighted CV)")
    for x, y, label in ((.02, .96, "2"), (.63, .96, "4"), (.02, .04, "1"), (.63, .04, "3")):
        ax.text(x, y, label, transform=ax.transAxes, color=PROFILE_COLORS[int(label)], fontsize=15, fontweight="bold",
                va="top" if y > .5 else "bottom")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(.06, .91), ncol=3)
    ax.set_title("a  "+("外形与内部结构的对应" if cn else "Outlines and internal organization"), loc="left", pad=20)
    ax = axes[1]
    values = counts.set_index("cluster")[[f"profile_{i}" for i in range(5)]].to_numpy()
    percentages = 100*values/values.sum(axis=1, keepdims=True)
    ax.imshow(percentages, cmap="Greys", vmin=0, vmax=75, aspect="auto")
    for row in range(3):
        for col in range(5):
            ax.text(col, row, f"{values[row, col]}\n{percentages[row, col]:.0f}%", ha="center", va="center",
                    color="white" if percentages[row, col] > 48 else "#263B42", fontsize=10)
    heat_labels = [names[i] if cn else names[i].replace(" paths", "\npaths") for i in range(5)]
    ax.set(xticks=range(5), xticklabels=heat_labels, yticks=range(3), yticklabels=shapes)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("b  "+("每种外形内部都包含不同组合" if cn else "Structural combinations within each outline"), loc="left", pad=20)
    fig.suptitle("三种外形之内，河网怎样组织汇合与路径？" if cn else "How are junctions and paths organized within the three river outlines?",
                 x=.035, ha="left", fontsize=16, y=.995)
    fig.text(.035, .025, (f"322 个不同的真实河网；只用几何数据划分。中位分界：汇合均衡度 {config['balance_cut']:.3f}，路径 CV {config['path_cut']:.3f}。\n"
        "“较均衡／较集中”均相对于本河网样本；两条无汇合河网单列。b 中百分比的分母是该外形的河网总数。") if cn else
        (f"322 distinct real networks; geometry-only cuts at median junction balance {config['balance_cut']:.3f} and path CV {config['path_cut']:.3f}.\n"
         "More/less and concentrated/dispersed are relative to this cohort. Two junction-free networks are separate. b: percentages within each outline."), fontsize=8)
    fig.subplots_adjust(left=.07, right=.98, top=.76, bottom=.23, wspace=.62)
    save(fig, "outline_and_internal_profiles")

    selected = reps.merge(networks, on=["comid", "station", "cluster", "profile"], validate="one_to_one")
    connection = sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True)
    line_hashes = {}
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 10.5), gridspec_kw={"height_ratios": [1.5, .8]})
    end = float(traces.relative_time.max())
    for row in selected.sort_values("profile").itertuples():
        i, color = int(row.profile)-1, PROFILE_COLORS[int(row.profile)]
        ax = axes[0, i]
        with np.load(CACHE/"members_full"/f"comid_{int(row.comid)}.npz") as member:
            line_hashes[str(row.comid)] = geometry_digest(connection, member["comids"])
        draw_network(ax, connection, row, color)
        n = int(networks.profile.eq(row.profile).sum())
        ax.set_title(f"{row.profile}  {names[row.profile]}\n{n} "+("个河网" if cn else "networks"), loc="left", fontsize=12, pad=12)
        detail = ((f"站点 {row.station} · {shapes[int(row.cluster)-1]}\n汇合 {row.tributary_balance:.3f} · 路径 CV {row.path_cv:.3f}\n"
            f"面积 {row.basin_area_km2:,.0f} km² · {int(row.n_reaches):,} 河段") if cn else
            (f"Station {row.station} · outline {int(row.cluster)}\nBalance {row.tributary_balance:.3f} · path CV {row.path_cv:.3f}\n"
             f"Area {row.basin_area_km2:,.0f} km² · {int(row.n_reaches):,} reaches"))
        ax.text(.02, -.10, detail, transform=ax.transAxes, va="top", fontsize=8.5)
        trace = traces[traces.profile.eq(row.profile)]
        ax = axes[1, i]
        for scenario, label, c, style in (("actual_spread", "实际路径" if cn else "Actual paths", color, "-"),
            ("half_spread", "路径差异减半" if cn else "Half spread", color, ":"),
            ("zero_spread", "路径完全相同" if cn else "Equal paths", "#AAB6B9", "--")):
            s = trace[trace.scenario.eq(scenario)]
            ax.plot(s.relative_time, s.outlet_anomaly, color=c, linestyle=style, linewidth=1.5, label=label)
        ax.set(xlim=(-.3, end), ylim=(0, 1.05), xlabel="相对传播时间（平均为 1）" if cn else "Relative time (mean delay = 1)")
        if i == 0:
            ax.set_ylabel("出口 DOC 脉冲 / 输入峰值" if cn else "Outlet anomaly / input peak")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.set_yticklabels([])
    connection.close()
    fig.suptitle("真实河网的四种内部结构组合" if cn else "Four internal structural profiles in actual mapped river networks",
                 x=.035, ha="left", fontsize=17, y=.995)
    fig.text(.035, .025, ("上排为按几何指标选出的真实代表，深色为主干、橙点为出口；各图比例尺独立。下排使用同一输入脉冲，并将平均传播距离统一。\n"
        "仅改变路径长短差异；没有降低输入 DOC 或加入沿程去除。路径时间为模型情景单位，不能读成天数或流速。") if cn else
        ("Top: geometry-only real medoids; dark mainstem, orange outlet, independent physical scale bars. Bottom: identical input with mean path delay fixed.\n"
         "Only path spread changes; no reduced input DOC or channel removal. Relative scenario time is not measured days or velocity."), fontsize=8)
    fig.subplots_adjust(left=.065, right=.985, top=.89, bottom=.13, wspace=.22, hspace=.58)
    save(fig, "real_internal_structure_atlas")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4))
    ax = axes[0]
    for p in range(1, 5):
        s = routing[routing.profile.eq(p)].set_index("scenario")
        ax.bar(p-.16, s.loc["actual_spread", "mean_peak"], width=.29, color=PROFILE_COLORS[p])
        ax.bar(p+.16, s.loc["half_spread", "mean_peak"], width=.29, color=PROFILE_COLORS[p], alpha=.4)
    ax.axhline(1, color="#AAB6B9", linestyle="--", linewidth=1.)
    ax.set(xticks=range(1, 5), xticklabels=heat_labels[1:], ylim=(0, 1.06),
        ylabel="平均出口峰值 / 相同输入峰值" if cn else "Mean outlet peak / identical input peak")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("a  "+("统一平均路径后的结构情景" if cn else "Routing with the same mean path delay"), loc="left", pad=18)
    ax.text(.03, .91, "实色：实际路径；浅色：长短差异减半" if cn else "Solid: actual paths; pale: half path spread", transform=ax.transAxes, fontsize=8)
    ax.text(.03, .83, "虚线：所有路径等长" if cn else "Dashed: all paths equal", transform=ax.transAxes, fontsize=8)
    ax = axes[1]
    labels = []
    for p in range(1, 5):
        for offset, metric, marker in ((-.09, "outlet_mixture_log_sd_ratio", "o"),
            (.09, "logscale_outlet_mixture_log_sd_ratio", "s")):
            r = observed[observed.profile.eq(p) & observed.metric.eq(metric)].iloc[0]
            if np.isfinite(r.ci_low):
                ax.plot(np.exp([r.ci_low, r.ci_high]), [p-1+offset]*2, color=PROFILE_COLORS[p], linewidth=1.5)
            ax.scatter(np.exp(r.estimate), p-1+offset, color=PROFILE_COLORS[p], s=35, marker=marker)
        labels.append(names[p].replace("\n", " / ")+f"\n{int(r.n_receivers)} "+("站" if cn else "sites")+f" / {int(r.n_blocks)} "+("系统" if cn else "systems"))
    ax.axvline(1, color="#9FAFB4", linestyle="--", linewidth=.9)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(yticks=range(4), yticklabels=labels, ylim=(3.6, -.6), xticks=[.2, .5, 1, 2, 5], xticklabels=["0.2", "0.5", "1", "2", "5"],
        xlabel="下游 / 被监测两支流混合的波动 SD 比" if cn else "Outlet / gauged-pair mixture SD ratio")
    ax.set_title("b  "+("实测 DOC：并不完全按情景排序" if cn else "Observed DOC does not follow a universal ordering"), loc="left", pad=18)
    fig.suptitle("内部结构的缓冲潜力与实测 DOC" if cn else "Internal structure: routing potential and observed DOC variability",
                 x=.035, ha="left", fontsize=16, y=.995)
    fig.text(.035, .025, ("a 为 320 个有汇合点河网的固定输入计算，峰值下降不等于 DOC 总量减少。b 为 22 个接收站的几何均值，● 原浓度；■ log1p。\n"
        "b 线为 5,000 次完整监测系统 bootstrap 95% 区间；监测系统可跨多个结构组。河网组的流域面积和监测覆盖不同，关联调整另表报告。") if cn else
        ("a: fixed-input scenarios in 320 networks with junctions; lower peaks do not mean less total DOC. b: 22 receivers; ● native concentration, ■ log1p.\n"
         "b: 5,000 whole monitoring-system bootstrap 95%; systems span profiles. Basin size and monitoring coverage differ; adjusted associations are reported separately."), fontsize=8)
    fig.subplots_adjust(left=.07, right=.985, top=.78, bottom=.24, wspace=.85)
    save(fig, "controlled_and_observed_buffers")
    # The full network and a gauged tributary pair represent different scales.
    receivers = pd.read_csv(a/"observed_receivers.csv", dtype={"target": str})
    alignment = pd.read_csv(a/"whole_pair_alignment.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))
    pairs = (("tributary_balance", "gauged_pair_minor_share"), ("path_cv", "gauged_pair_path_cv"))
    for i, (ax, (whole, pair)) in enumerate(zip(axes, pairs, strict=True)):
        for p in range(1, 5):
            s = receivers[receivers.profile.eq(p)]
            ax.scatter(s[whole], s[pair], color=PROFILE_COLORS[p], s=45, label=f"{p}: "+names[p].replace("\n", " / "))
        rho = alignment[alignment.whole_network_descriptor.eq(whole)].iloc[0].spearman
        ax.text(.98, .04, f"Spearman ρ = {rho:.2f}\n22 "+("接收站 · 11 监测系统" if cn else "receivers · 11 systems"),
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9)
        ax.set_xlabel(("整条河网的典型汇合均衡度" if i == 0 else "整条河网的路径 CV") if cn else
                      ("Whole-network median junction balance" if i == 0 else "Whole-network path CV"))
        ax.set_ylabel(("被监测支流对的小支流面积份额" if i == 0 else "被监测两条支流的路径 CV") if cn else
                      ("Gauged pair: smaller branch area share" if i == 0 else "Gauged pair: two-path CV"))
        ax.set_title(("a  汇合组织" if i == 0 else "b  路径差异") if cn else
                     ("a  Junction organization" if i == 0 else "b  Path dispersion"), loc="left", pad=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower left", bbox_to_anchor=(.06, .10), ncol=2, fontsize=8)
    fig.suptitle("整条河网与监测支流：测到的是同一结构吗？" if cn else "Whole networks and gauged pairs: do the structural measurements align?",
                 x=.035, ha="left", fontsize=15, y=.995)
    fig.text(.035, .025, "每点为一个接收站，支流对指标先在站内平均。相关只描述尺度对应，不是 DOC 效应。" if cn else
             "One point per receiver; pair descriptors averaged within receiver. Correlations describe scale alignment, not DOC effects.", fontsize=8)
    fig.subplots_adjust(left=.08, right=.97, top=.81, bottom=.30, wspace=.34)
    save(fig, "measurement_scale_alignment")
    sources += [a/"observed_receivers.csv", a/"whole_pair_alignment.csv"]
    sources += [Path("scripts/plot_doc_river_planform_v1.py"), Path("src/river_graph/topology/river_planform.py")]
    for row in selected.itertuples():
        sources += [CACHE/"members_full"/f"comid_{row.comid}.npz", CACHE/"basins"/f"comid_{row.comid}.json"]
    manifest = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in sources},
        "representative_flowline_hashes": line_hashes,
        "generator_sha256": sha256_file(Path(__file__)), "figure_hashes": {str(p): sha256_file(p) for p in written}}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
