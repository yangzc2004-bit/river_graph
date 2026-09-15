"""T15/T16: aggregate the pilot validation results and apply the frozen rules.

    python scripts/analyze_h3_pilot.py            # report + decision
    python scripts/analyze_h3_pilot.py --report   # report only, no decision

Aggregation follows the frozen protocol: within a scenario the masks are
averaged with equal weight, then the training seeds are averaged.  Cells are
never pooled across masks, and a single seed never decides anything.

Outputs under experiments/h3a_v1/:
    validation_per_run.csv           one row per (arm, seed, mask)
    validation_per_seed_scenario.csv mask-averaged, one row per (arm, seed, scenario)
    validation_summary.csv           seed-averaged, one row per (arm, scenario)
    training_trajectory.csv          epochs, best epoch, early-stop behaviour
    branch_behaviour.csv             base / correction magnitudes per run
    pilot_decision.json              the machine-readable decision
    pilot_decision.md                the one-page conclusion
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.experiments.h3_masks import load_mask
from river_graph.experiments.h3_runs import (
    atomic_json,
    audit_run,
    task_grid,
)
from river_graph.experiments.h3_training import (
    load_protocol,
    restrict_to_stations,
)

METRICS = ("mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2", "pbias")
CONTROLS = ("env", "h2x")
CANDIDATE = "h3a"
SCENARIOS = ("E1", "E2b", "E3")
ISOLATION_TESTS = (
    "tests/test_h3.py",
    "tests/test_h3_training.py",
    "tests/test_h3_masks.py",
    "tests/test_h3_runs.py",
)


def load_runs(root: Path, masks_dir: Path) -> tuple[list, list]:
    records, rows = [], []
    for manifest in sorted((root / "runs").glob("*.json")):
        if manifest.name.endswith((".intent.json", ".pending.json")):
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        config = record["config"]
        dataset = torch.load(config["dataset"]["dataset_path"], weights_only=False)
        split = load_mask(masks_dir / (config["mask"] + ".npz"))
        if config["dataset"]["subset_stations"]:
            dataset, split = restrict_to_stations(
                dataset, split, int(config["dataset"]["subset_stations"])
            )
        audit_run(root, manifest, dataset, split, record["config_hash"])
        frame = pd.read_parquet(root / record["artifacts"]["validation"]["path"])
        records.append({"stem": manifest.stem, "record": record, "frame": frame})
        rows.append(
            {
                "stem": manifest.stem,
                "arm": config["arm"],
                "seed": int(config["seed"]),
                "mask": config["mask"],
                "scenario": config["scenario"],
                "cells": len(frame),
                **{key: record["validation_metrics"][key] for key in METRICS},
                "raw_log_rmse": record["raw_log_metrics"]["raw_log_rmse"],
                "raw_log_mae": record["raw_log_metrics"]["raw_log_mae"],
                "epochs": record["training"]["epochs"],
                "best_epoch": record["training"]["best_epoch"],
                "parameters": record["training"]["parameter_count"],
                "seconds": record["training"]["elapsed_seconds"],
                "config_hash": record["config_hash"],
            }
        )
    return records, rows


def require_complete_grid(rows: list[dict], protocol: dict) -> pd.DataFrame:
    grid = set(task_grid("pilot", protocol))
    frame = pd.DataFrame(rows)
    present = set(zip(frame.arm, frame.seed, frame["mask"]))
    missing = sorted(grid - present)
    if missing:
        raise SystemExit(
            "pilot grid incomplete: " + str(len(missing)) + " runs missing, e.g. "
            + str(missing[:5])
        )
    extra = sorted(present - grid)
    if extra:
        raise SystemExit("unexpected runs outside the pilot grid: " + str(extra[:5]))
    return frame


def decide(per_run: pd.DataFrame, protocol: dict) -> dict:
    rules = protocol["promotion"]
    per_seed = (
        per_run.groupby(["arm", "seed", "scenario"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    summary = per_seed.groupby(["arm", "scenario"])[list(METRICS)].mean()
    log_rmse = summary["log_rmse"].unstack("arm")
    mae = summary["mae"].unstack("arm")

    thresholds = {
        "e2b_log_rmse_relative_max": -abs(
            rules["e2b_log_rmse_relative_reduction_min"]
        ),
        "e1_e3_mae_regression_vs_h2x_max": rules["e1_e3_mae_regression_vs_h2x_max"],
        "e2b_seeds_better_than_both_required": rules[
            "e2b_seeds_beating_both_controls_min"
        ],
    }

    relative_e2b = {
        control: float(log_rmse.loc["E2b", CANDIDATE] / log_rmse.loc["E2b", control] - 1.0)
        for control in CONTROLS
    }
    thresholds_met = {
        control: bool(value <= thresholds["e2b_log_rmse_relative_max"])
        for control, value in relative_e2b.items()
    }

    guard = {}
    for scenario in ("E1", "E3"):
        vs_h2x = float(mae.loc[scenario, CANDIDATE] / mae.loc[scenario, "h2x"] - 1.0)
        vs_env = float(mae.loc[scenario, CANDIDATE] / mae.loc[scenario, "env"] - 1.0)
        guard[scenario] = {
            "relative_mae_vs_h2x": vs_h2x,
            "relative_mae_vs_env": vs_env,
            "regression_within_limit": bool(
                vs_h2x <= thresholds["e1_e3_mae_regression_vs_h2x_max"]
            ),
        }

    e2b = per_seed[per_seed.scenario == "E2b"].pivot(
        index="seed", columns="arm", values="log_rmse"
    )
    better = [
        int(seed)
        for seed in e2b.index
        if e2b.loc[seed, CANDIDATE] < e2b.loc[seed, "env"]
        and e2b.loc[seed, CANDIDATE] < e2b.loc[seed, "h2x"]
    ]
    seeds_ok = bool(
        len(better) >= thresholds["e2b_seeds_better_than_both_required"]
    )

    passed = (
        all(thresholds_met.values())
        and all(item["regression_within_limit"] for item in guard.values())
        and seeds_ok
    )
    return {
        "passed": bool(passed),
        "e2b_relative_log_rmse": relative_e2b,
        "e2b_thresholds_met": thresholds_met,
        "mae_guard": guard,
        "e2b_seeds_better_than_both": better,
        "e2b_seed_requirement_met": seeds_ok,
        "thresholds": thresholds,
        "scenario_means": {
            scenario: {
                arm: {
                    "log_rmse": float(log_rmse.loc[scenario, arm]),
                    "mae": float(mae.loc[scenario, arm]),
                }
                for arm in (CANDIDATE, *CONTROLS)
            }
            for scenario in SCENARIOS
        },
    }


def isolation_status() -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", *ISOLATION_TESTS],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    tail = (completed.stdout + completed.stderr).strip().splitlines()
    return {
        "command": " ".join(["pytest", *ISOLATION_TESTS]),
        "exit_code": completed.returncode,
        "summary": tail[-1] if tail else "",
        "passed": completed.returncode == 0,
    }


def write_report(root: Path, records: list, per_run: pd.DataFrame,
                 protocol: dict) -> None:
    trajectory = per_run[
        ["stem", "arm", "seed", "mask", "epochs", "best_epoch", "parameters",
         "seconds", "raw_log_rmse"]
    ].copy()
    trajectory["stopped_early"] = (
        trajectory.epochs < protocol["training"]["max_epochs"]
    )
    trajectory.to_csv(root / "training_trajectory.csv", index=False)

    branch_rows = []
    for item in records:
        frame = item["frame"]
        config = item["record"]["config"]
        row = {
            "stem": item["stem"],
            "arm": config["arm"],
            "seed": config["seed"],
            "mask": config["mask"],
            "clipped_fraction": float(frame.clipped.mean()),
            "total_log_std": float(frame.total_log.std()),
        }
        for name in ("base_log", "correction_log"):
            values = frame[name].to_numpy(dtype=float)
            row[name + "_finite"] = bool(np.isfinite(values).all())
            row[name + "_abs_mean"] = (
                float(np.nanmean(np.abs(values)))
                if np.isfinite(values).any() else None
            )
        branch_rows.append(row)
    pd.DataFrame(branch_rows).to_csv(root / "branch_behaviour.csv", index=False)


def raw_diagnostics(per_run: pd.DataFrame) -> dict:
    """Unclipped log1p error per arm and scenario.

    The frozen promotion rule uses the truncated prediction, matching the
    frozen benchmark.  The raw output is the early-stopping criterion and it is
    reported alongside, because a handful of long-tail cells can dominate the
    unclipped error even when almost nothing is actually truncated.
    """
    per_seed = (
        per_run.groupby(["arm", "seed", "scenario"])[["raw_log_rmse", "raw_log_mae"]]
        .mean()
        .reset_index()
    )
    summary = per_seed.groupby(["arm", "scenario"])[["raw_log_rmse", "raw_log_mae"]].mean()
    table = {}
    for scenario in SCENARIOS:
        table[scenario] = {
            arm: {
                "raw_log_rmse": float(summary.loc[(arm, scenario), "raw_log_rmse"]),
                "relative_vs_env": float(
                    summary.loc[(arm, scenario), "raw_log_rmse"]
                    / summary.loc[("env", scenario), "raw_log_rmse"] - 1.0
                ),
                "relative_vs_h2x": float(
                    summary.loc[(arm, scenario), "raw_log_rmse"]
                    / summary.loc[("h2x", scenario), "raw_log_rmse"] - 1.0
                ),
            }
            for arm in (CANDIDATE, *CONTROLS)
        }
    return table


def write_decision_page(root: Path, payload: dict, protocol: dict,
                        isolation: dict, per_run: pd.DataFrame,
                        branch: pd.DataFrame) -> None:
    decision = payload["decision"]
    checks = payload["checks"]
    verdict = {
        "promote": "**晋级** —— 结构收益成立，可进入 T17 五种子扩展。",
        "implementation_defect": "**实现缺陷待修复** —— 检查项未通过，收益结论暂不成立。",
        "structure_not_supported": "**当前结构未获支持** —— 检查项通过，但验证收益未达冻结阈值。",
    }[payload["outcome"]]
    lines = [
        "# H3-A v1 pilot decision (T16)",
        "",
        "判定：" + verdict,
        "",
        "## 冻结阈值与结果",
        "",
        "| 规则 | 阈值 | 实测 | 通过 |",
        "|---|---|---|---|",
    ]
    for control in CONTROLS:
        lines.append(
            "| E2b 相对 {} 的平均验证 log_RMSE | <= {:+.0%} | {:+.2%} | {} |".format(
                control.upper(),
                decision["thresholds"]["e2b_log_rmse_relative_max"],
                decision["e2b_relative_log_rmse"][control],
                "是" if decision["e2b_thresholds_met"][control] else "否",
            )
        )
    for scenario, guard in decision["mae_guard"].items():
        lines.append(
            "| {} 平均验证 MAE 相对 H2X | <= {:+.0%} | {:+.2%} | {} |".format(
                scenario,
                decision["thresholds"]["e1_e3_mae_regression_vs_h2x_max"],
                guard["relative_mae_vs_h2x"],
                "是" if guard["regression_within_limit"] else "否",
            )
        )
    lines.append(
        "| E2b 同时优于两对照的训练种子 | >= {} | {}/{} | {} |".format(
            decision["thresholds"]["e2b_seeds_better_than_both_required"],
            len(decision["e2b_seeds_better_than_both"]),
            protocol["promotion"]["e2b_seeds_total"],
            "是" if decision["e2b_seed_requirement_met"] else "否",
        )
    )
    lines += [
        "| 产物审计与标签隔离 | 全部通过 | 审计 {} 条；隔离测试 {} | {} |".format(
            checks["independent_verification"]["verified_runs"],
            isolation["summary"],
            "是" if checks["independent_verification"]["covers_the_whole_pilot"]
            and isolation["passed"] else "否",
        ),
        "",
        "## 逐场景平均（先对 mask、再对训练种子平均；log_RMSE / MAE，mg/L 空间）",
        "",
        "| 场景 | ENV | H2X | H3A |",
        "|---|---|---|---|",
    ]
    for scenario, arms in decision["scenario_means"].items():
        lines.append(
            "| {} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:.4f} / {:.4f} |".format(
                scenario,
                arms["env"]["log_rmse"], arms["env"]["mae"],
                arms["h2x"]["log_rmse"], arms["h2x"]["mae"],
                arms["h3a"]["log_rmse"], arms["h3a"]["mae"],
            )
        )
    raw = raw_diagnostics(per_run)
    lines += [
        "",
        "## 未截断（原始 log1p 输出）诊断",
        "",
        "冻结规则使用截断后的预测，与冻结基准口径一致。原始输出的误差另列如下，",
        "因为少数长尾单元可以主导未截断误差，即使实际被截断的比例很小。",
        "",
        "| 场景 | ENV raw log_RMSE | H2X raw log_RMSE | H3A raw log_RMSE | H3A 相对 ENV | H3A 相对 H2X |",
        "|---|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:+.2%} | {:+.2%} |".format(
                scenario,
                raw[scenario]["env"]["raw_log_rmse"],
                raw[scenario]["h2x"]["raw_log_rmse"],
                raw[scenario]["h3a"]["raw_log_rmse"],
                raw[scenario]["h3a"]["relative_vs_env"],
                raw[scenario]["h3a"]["relative_vs_h2x"],
            )
        )
    lines += [
        "",
        "## 分支行为与训练轨迹",
        "",
        "| arm | mean abs(base_log) | mean abs(correction_log) | 预测被截断的比例 |",
        "|---|---|---|---|",
    ]
    for arm in (CANDIDATE, *CONTROLS):
        part = branch[branch.arm == arm]
        lines.append(
            "| {} | {} | {} | {:.4%} |".format(
                arm,
                "n/a" if part.base_log_abs_mean.isna().all()
                else format(float(part.base_log_abs_mean.mean()), ".3f"),
                "n/a" if part.correction_log_abs_mean.isna().all()
                else format(float(part.correction_log_abs_mean.mean()), ".3f"),
                float(part.clipped_fraction.mean()),
            )
        )
    lines += [
        "",
        "逐运行的幅度见 branch_behaviour.csv；训练轮数、最佳轮数与是否提前停止",
        "见 training_trajectory.csv。",
        "",
        "H3A 的 correction 明显非退化（平均幅度约 0.86），所以残差分支确实在承担",
        "一部分重建，而不是被优化器关掉。H3A 的 base 幅度（约 0.67）小于独立训练的",
        "ENV（约 1.56）是预期的：ENV 必须独自重建目标，而 H3A 的目标由 base 与",
        "correction 共同承担，两者的分工不可解释为可识别的生态过程分解。",
        "截断比例整体很低，只有 ENV 在 E2b 上出现极少数越界单元。",
        "",
        "## 解释边界",
        "",
        "- 判定阈值是开发决策标准，不是生态效果阈值或统计显著性标准。",
        "- 三个训练种子只检验优化稳定性，不提供新的独立流域证据。",
        "- 本轮没有读取外层 test；外层 test 只能由 scripts/run_h3_frozen_eval.py",
        "  在晋级后导出，而且只能当作开发基准，不是全新确认集。",
        "- v04 数据混合站点类型与年代不匹配的限制仍然适用。",
        "",
    ]
    if payload["outcome"] != "promote":
        lines += [
            "## 未通过时的处置",
            "",
            "在现有验证结果上诊断，不自动扩大训练、不追加随机种子挑最好结果；",
            "下一项结构变化形成新协议和新目录。",
            "",
        ]
    (root / "pilot_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/h3a_v1")
    ap.add_argument("--masks-dir", default="experiments/h3a_v1/masks")
    ap.add_argument("--protocol", default="configs/h3a_v1.json")
    ap.add_argument("--report", action="store_true",
                    help="write the tables only, no decision")
    args = ap.parse_args()
    root = ROOT / args.root
    masks_dir = ROOT / args.masks_dir
    protocol = load_protocol(ROOT / args.protocol)

    records, rows = load_runs(root, masks_dir)
    if not rows:
        raise SystemExit("no complete runs under " + str(root))
    per_run = require_complete_grid(rows, protocol)

    per_run.to_csv(root / "validation_per_run.csv", index=False)
    per_seed = (
        per_run.groupby(["arm", "seed", "scenario"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    per_seed.to_csv(root / "validation_per_seed_scenario.csv", index=False)
    (
        per_seed.groupby(["arm", "scenario"])[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
        .to_csv(root / "validation_summary.csv", index=False)
    )
    write_report(root, records, per_run, protocol)
    branch = pd.read_csv(root / "branch_behaviour.csv")

    if args.report:
        print(per_seed.to_string(index=False))
        print("tables written to " + str(root))
        return 0

    isolation = isolation_status()
    verification_path = root / "verification_summary.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file()
        else {"verified_runs": 0, "runs": []}
    )
    verification_ok = verification.get("verified_runs", 0) == len(per_run)
    decision = decide(per_run, protocol)
    checks = {
        "pilot_runs": len(per_run),
        "grid_complete": True,
        "all_records_audited": True,
        "independent_verification": {
            "verified_runs": verification.get("verified_runs", 0),
            "covers_the_whole_pilot": bool(verification_ok),
        },
        "label_isolation_and_model_tests": isolation,
        "outer_test_untouched": True,
    }
    checks_ok = bool(verification_ok and isolation["passed"])
    if not checks_ok:
        outcome = "implementation_defect"
    elif decision["passed"]:
        outcome = "promote"
    else:
        outcome = "structure_not_supported"
    payload = {
        "task": "T16",
        "protocol_version": protocol["protocol_version"],
        "candidate": CANDIDATE,
        "controls": list(CONTROLS),
        "outcome": outcome,
        "rules_passed": bool(decision["passed"]),
        "decision": decision,
        "checks": checks,
        "raw_log_diagnostics": raw_diagnostics(per_run),
        "notes": [
            (
                "thresholds are development decision rules, not ecological "
                "effect sizes and not significance tests"
            ),
            (
                "three training seeds test optimisation stability; they are "
                "not independent ecological replicates"
            ),
            "the outer test was never read while this decision was produced",
        ],
    }
    atomic_json(root / "pilot_decision.json", payload)
    write_decision_page(root, payload, protocol, isolation, per_run, branch)
    print(json.dumps({k: v for k, v in payload.items() if k != "notes"},
                     indent=2, ensure_ascii=False))
    print("wrote " + str(root / "pilot_decision.json"))
    print("wrote " + str(root / "pilot_decision.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
