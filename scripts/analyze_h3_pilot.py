"""T15/T16/T17: aggregate validation results and apply the frozen rules.

    python scripts/analyze_h3_pilot.py --stage pilot             # T16 decision
    python scripts/analyze_h3_pilot.py --stage expand            # T17 decision
    python scripts/analyze_h3_pilot.py --stage pilot --report    # tables only

Aggregation follows the frozen protocol: within a scenario the masks are
averaged with equal weight, then the training seeds are averaged.  Cells are
never pooled across masks, and a single seed never decides anything.

Everything reported about truncation is computed from the per-run prediction
tables.  No statement about which arm is clipped, or about how truncation
changes a comparison, is written into this script as a constant.

Outputs under the stage root:
    validation_per_run.csv           one row per (arm, seed, mask)
    validation_per_seed_scenario.csv mask-averaged, one row per (arm, seed, scenario)
    validation_summary.csv           seed-averaged, one row per (arm, scenario)
    training_trajectory.csv          epochs, best epoch, early-stop behaviour
    branch_behaviour.csv             base / correction magnitudes and clipping
    clipping_report.json             truncation counts, fractions and effects
    pilot_decision.json / expansion_decision.json   the machine-readable decision
    pilot_decision.md / expansion_decision.md       the one-page conclusion
"""

from __future__ import annotations

import argparse
import hashlib
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
    expected_config_for,
    task_grid,
)
from river_graph.experiments.h3_training import (
    load_protocol,
    restrict_to_stations,
)
from river_graph.experiments.provenance import sha256_file

METRICS = ("mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2", "pbias")
CONTROLS = ("env", "h2x")
CANDIDATE = "h3a"
GUARD_SCENARIOS = ("E1", "E3")
PRIMARY_SCENARIO = "E2b"
ISOLATION_TESTS = (
    "tests/test_h3.py",
    "tests/test_h3_training.py",
    "tests/test_h3_masks.py",
    "tests/test_h3_runs.py",
)
STAGE_RESULT = {
    "pilot": ("pilot_decision", "T16"),
    "expand": ("expansion_decision", "T17"),
}


def load_runs(root: Path, masks_dir: Path, protocol: dict) -> tuple[list, list]:
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
        expected, expected_hash = expected_config_for(record, protocol, masks_dir)
        audit_run(
            root, manifest, dataset, split, expected_hash=expected_hash,
            expected_config=expected, strict_identity=True,
        )
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
                "clipped_cells": int(frame["clipped"].sum()),
                "epochs": record["training"]["epochs"],
                "best_epoch": record["training"]["best_epoch"],
                "parameters": record["training"]["parameter_count"],
                "seconds": record["training"]["elapsed_seconds"],
                "config_hash": record["config_hash"],
            }
        )
    return records, rows


def require_complete_grid(rows: list[dict], protocol: dict, stage: str) -> pd.DataFrame:
    grid = set(task_grid(stage, protocol))
    frame = pd.DataFrame(rows)
    present = set(zip(frame.arm, frame.seed, frame["mask"]))
    missing = sorted(grid - present)
    if missing:
        raise SystemExit(
            stage + " grid incomplete: " + str(len(missing)) + " runs missing, e.g. "
            + str(missing[:5])
        )
    extra = sorted(present - grid)
    if extra:
        raise SystemExit("runs outside the " + stage + " grid: " + str(extra[:5]))
    return frame


def artifact_manifest(records: list, root: Path) -> tuple[list, str]:
    """Identity of the exact runs a decision was taken on."""
    entries = []
    for item in sorted(records, key=lambda r: r["stem"]):
        record = item["record"]
        entries.append(
            {
                "stem": item["stem"],
                "arm": record["config"]["arm"],
                "seed": record["config"]["seed"],
                "mask": record["config"]["mask"],
                "config_hash": record["config_hash"],
                "checkpoint_sha256": sha256_file(
                    root / record["artifacts"]["checkpoint"]["path"]
                ),
            }
        )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return entries, digest


def clipping_report(per_run: pd.DataFrame, per_seed: pd.DataFrame) -> dict:
    """Truncation facts, computed from the runs; nothing here is hard-coded."""
    per_arm = {}
    for arm, part in per_run.groupby("arm"):
        clipped = int(part.clipped_cells.sum())
        cells = int(part.cells.sum())
        raw = float(part.raw_log_rmse.mean())
        clipped_rmse = float(part.log_rmse.mean())
        per_arm[arm] = {
            "clipped_cells": clipped,
            "cells": cells,
            "clipped_fraction": clipped / cells if cells else 0.0,
            "runs_with_clipping": int((part.clipped_cells > 0).sum()),
            "runs": len(part),
            "mean_raw_log_rmse": raw,
            "mean_clipped_log_rmse": clipped_rmse,
            "truncation_effect": raw - clipped_rmse,
        }

    # per_seed only carries the headline metrics, so aggregate the raw and
    # truncated errors here from the per-run table, within scenario first.
    scored = (
        per_run.groupby(["arm", "seed", "scenario"])[["log_rmse", "raw_log_rmse"]]
        .mean()
        .reset_index()
    )
    e2b = scored[scored.scenario == PRIMARY_SCENARIO]
    grid = e2b.pivot(index="seed", columns="arm", values="log_rmse")
    raw_grid = e2b.pivot(index="seed", columns="arm", values="raw_log_rmse")
    def better(grid_):
        return [
            int(seed)
            for seed in grid_.index
            if grid_.loc[seed, CANDIDATE] < grid_.loc[seed, "env"]
            and grid_.loc[seed, CANDIDATE] < grid_.loc[seed, "h2x"]
        ]
    effects = {}
    for arm in (CANDIDATE, *CONTROLS):
        raw_mean = float(
            e2b[e2b.arm == arm].raw_log_rmse.mean()
        )
        clipped_mean = float(e2b[e2b.arm == arm].log_rmse.mean())
        effects[arm] = {"raw": raw_mean, "clipped": clipped_mean}
    relative = {}
    for control in CONTROLS:
        raw_rel = effects[CANDIDATE]["raw"] / effects[control]["raw"] - 1.0
        clipped_rel = effects[CANDIDATE]["clipped"] / effects[control]["clipped"] - 1.0
        relative[control] = {
            "raw_relative": float(raw_rel),
            "clipped_relative": float(clipped_rel),
            "truncation_moved_advantage_by": float(clipped_rel - raw_rel),
        }
    return {
        "per_arm": per_arm,
        "e2b_raw_vs_clipped": effects,
        "e2b_relative": relative,
        "e2b_seeds_better_than_both_raw": better(raw_grid),
        "e2b_seeds_better_than_both_clipped": better(grid),
        "note": (
            "The frozen promotion rule uses the truncated prediction, matching "
            "the frozen benchmark. These are the same numbers untruncated, and "
            "the change in the comparison they imply."
        ),
    }


def decide(per_run: pd.DataFrame, protocol: dict, stage: str) -> dict:
    rules = protocol["promotion"]
    expand = stage == "expand"
    thresholds = (
        rules["expand"] if expand
        else {
            "e2b_log_rmse_relative_reduction_min": rules[
                "e2b_log_rmse_relative_reduction_min"
            ],
            "e1_e3_mae_regression_vs_h2x_max": rules[
                "e1_e3_mae_regression_vs_h2x_max"
            ],
            "e2b_seeds_beating_both_controls_min": rules[
                "e2b_seeds_beating_both_controls_min"
            ],
            "e2b_seeds_total": rules["e2b_seeds_total"],
        }
    )
    per_seed = (
        per_run.groupby(["arm", "seed", "scenario"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    summary = per_seed.groupby(["arm", "scenario"])[list(METRICS)].mean()
    log_rmse = summary["log_rmse"].unstack("arm")
    mae = summary["mae"].unstack("arm")
    scenarios = sorted(log_rmse.index)

    relative_primary = {
        control: float(
            log_rmse.loc[PRIMARY_SCENARIO, CANDIDATE]
            / log_rmse.loc[PRIMARY_SCENARIO, control] - 1.0
        )
        for control in CONTROLS
    }
    limit = -abs(thresholds["e2b_log_rmse_relative_reduction_min"])
    primary_ok = {
        control: bool(value <= limit) for control, value in relative_primary.items()
    }

    guard = {}
    for scenario in GUARD_SCENARIOS:
        if scenario not in log_rmse.index:
            continue
        vs_h2x = float(mae.loc[scenario, CANDIDATE] / mae.loc[scenario, "h2x"] - 1.0)
        vs_env = float(mae.loc[scenario, CANDIDATE] / mae.loc[scenario, "env"] - 1.0)
        guard[scenario] = {
            "relative_mae_vs_h2x": vs_h2x,
            "relative_mae_vs_env": vs_env,
            "regression_within_limit": bool(
                vs_h2x <= thresholds["e1_e3_mae_regression_vs_h2x_max"]
            ),
        }

    primary = per_seed[per_seed.scenario == PRIMARY_SCENARIO].pivot(
        index="seed", columns="arm", values="log_rmse"
    )
    better = [
        int(seed)
        for seed in primary.index
        if primary.loc[seed, CANDIDATE] < primary.loc[seed, "env"]
        and primary.loc[seed, CANDIDATE] < primary.loc[seed, "h2x"]
    ]
    seeds_ok = len(better) >= thresholds["e2b_seeds_beating_both_controls_min"]
    passed = (
        all(primary_ok.values())
        and all(item["regression_within_limit"] for item in guard.values())
        and seeds_ok
    )
    return {
        "stage": stage,
        "passed": bool(passed),
        "primary_scenario": PRIMARY_SCENARIO,
        "relative_log_rmse": relative_primary,
        "primary_thresholds_met": primary_ok,
        "mae_guard": guard,
        "seeds_better_than_both": better,
        "seed_requirement_met": bool(seeds_ok),
        "thresholds": {
            "primary_log_rmse_relative_max": limit,
            "guard_mae_regression_vs_h2x_max": thresholds[
                "e1_e3_mae_regression_vs_h2x_max"
            ],
            "seeds_better_than_both_required": thresholds[
                "e2b_seeds_beating_both_controls_min"
            ],
            "seeds_total": thresholds["e2b_seeds_total"],
        },
        "scenario_means": {
            scenario: {
                arm: {
                    "log_rmse": float(log_rmse.loc[scenario, arm]),
                    "mae": float(mae.loc[scenario, arm]),
                }
                for arm in (CANDIDATE, *CONTROLS)
            }
            for scenario in scenarios
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


def write_tables(root: Path, records: list, per_run: pd.DataFrame,
                 protocol: dict) -> pd.DataFrame:
    per_seed = (
        per_run.groupby(["arm", "seed", "scenario"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    per_run.to_csv(root / "validation_per_run.csv", index=False)
    per_seed.to_csv(root / "validation_per_seed_scenario.csv", index=False)
    (
        per_seed.groupby(["arm", "scenario"])[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
        .to_csv(root / "validation_summary.csv", index=False)
    )
    trajectory = per_run[
        ["stem", "arm", "seed", "mask", "epochs", "best_epoch", "parameters",
         "seconds", "raw_log_rmse"]
    ].copy()
    max_epochs = protocol["training"]["max_epochs"]
    patience = protocol["training"]["patience"]
    trajectory["stopped_early"] = trajectory.epochs < max_epochs
    trajectory["stopped_at_patience"] = (
        trajectory.epochs - trajectory.best_epoch
    ) >= patience
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
            "clipped_cells": int(frame.clipped.sum()),
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
    branch = pd.DataFrame(branch_rows)
    branch.to_csv(root / "branch_behaviour.csv", index=False)
    return per_seed


def write_decision_page(root: Path, payload: dict, decision: dict,
                        isolation: dict, stage: str, branch: pd.DataFrame,
                        clipping: dict) -> None:
    checks = payload["checks"]
    verdict = {
        "promote": "**晋级** —— 达到冻结的扩展标准。",
        "implementation_defect": "**实现缺陷待修复** —— 检查项未通过，收益结论暂不成立。",
        "structure_not_supported": "**当前结构未获支持** —— 检查项通过，但验证收益未达冻结阈值。",
    }[payload["outcome"]]
    title = "pilot decision (T16)" if stage == "pilot" else "expansion decision (T17)"
    lines = [
        "# H3-A v1 " + title,
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
            "| {} 相对 {} 的平均验证 log_RMSE | <= {:+.0%} | {:+.2%} | {} |".format(
                decision["primary_scenario"], control.upper(),
                decision["thresholds"]["primary_log_rmse_relative_max"],
                decision["relative_log_rmse"][control],
                "是" if decision["primary_thresholds_met"][control] else "否",
            )
        )
    for scenario, guard in decision["mae_guard"].items():
        lines.append(
            "| {} 平均验证 MAE 相对 H2X | <= {:+.0%} | {:+.2%} | {} |".format(
                scenario,
                decision["thresholds"]["guard_mae_regression_vs_h2x_max"],
                guard["relative_mae_vs_h2x"],
                "是" if guard["regression_within_limit"] else "否",
            )
        )
    lines.append(
        "| {} 同时优于两对照的训练种子 | >= {} | {}/{} | {} |".format(
            decision["primary_scenario"],
            decision["thresholds"]["seeds_better_than_both_required"],
            len(decision["seeds_better_than_both"]),
            decision["thresholds"]["seeds_total"],
            "是" if decision["seed_requirement_met"] else "否",
        )
    )
    lines += [
        "| 产物审计与标签隔离 | 全部通过 | 严格身份核验 {} 条；隔离测试 {} | {} |".format(
            checks["independent_verification"]["verified_runs"],
            isolation["summary"],
            "是" if checks["independent_verification"]["covers_the_whole_grid"]
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

    lines += [
        "",
        "## 截断（由逐运行预测动态生成）",
        "",
        "| arm | 被截断单元 | 占总单元 | 发生截断的运行 | 原始 log_RMSE | 截断后 log_RMSE | 截断影响 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in clipping["per_arm"].items():
        lines.append(
            "| {} | {} | {:.4%} | {}/{} | {:.4f} | {:.4f} | {:+.4f} |".format(
                arm, item["clipped_cells"], item["clipped_fraction"],
                item["runs_with_clipping"], item["runs"],
                item["mean_raw_log_rmse"], item["mean_clipped_log_rmse"],
                item["truncation_effect"],
            )
        )
    lines += [
        "",
        "E2b 逐种子同时优于两对照的种子数：原始输出 {}，截断后 {}。".format(
            clipping["e2b_seeds_better_than_both_raw"],
            clipping["e2b_seeds_better_than_both_clipped"],
        ),
        "截断把 H3A 相对 ENV 的优势改变了 {:+.2%}，相对 H2X 改变了 {:+.2%}".format(
            clipping["e2b_relative"]["env"]["truncation_moved_advantage_by"],
            clipping["e2b_relative"]["h2x"]["truncation_moved_advantage_by"],
        ),
        "（正值表示截断扩大了 H3A 的优势）。冻结规则使用截断后的预测；",
        "原始输出上的对照同时列出，供判读。",
        "",
        "## 分支行为与训练轨迹",
        "",
        "| arm | mean abs(base_log) | mean abs(correction_log) |",
        "|---|---|---|",
    ]
    for arm in (CANDIDATE, *CONTROLS):
        part = branch[branch.arm == arm]
        lines.append(
            "| {} | {} | {} |".format(
                arm,
                "n/a" if part.base_log_abs_mean.isna().all()
                else format(float(part.base_log_abs_mean.mean()), ".3f"),
                "n/a" if part.correction_log_abs_mean.isna().all()
                else format(float(part.correction_log_abs_mean.mean()), ".3f"),
            )
        )
    lines += [
        "",
        "逐运行的幅度与截断见 branch_behaviour.csv；训练轮数、最佳轮数、是否提前停止",
        "以及是否停在 patience 边界见 training_trajectory.csv。",
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
    lines += [
        "## 解释边界",
        "",
        "- 判定阈值是开发决策标准，不是生态效果阈值或统计显著性标准。",
        "- 训练种子只检验优化稳定性，不提供新的独立流域证据。",
        "- 本轮没有读取外层 test；外层 test 只能由 scripts/run_h3_frozen_eval.py",
        "  在扩展判定为 promote 且配置与产物清单完整时导出，而且只能当作开发基准。",
        "- v05 数据已应用冻结的协变量质量规则；DOC 标签本身未改动。",
        "- 混合站点类型与年代不匹配的限制仍然适用。",
        "",
    ]
    (root / (STAGE_RESULT[stage][0] + ".md")).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("pilot", "expand"), default="pilot")
    ap.add_argument("--root", default=None)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--protocol", default="configs/h3a_v1.json")
    ap.add_argument("--report", action="store_true",
                    help="write the tables only, no decision")
    args = ap.parse_args()
    protocol = load_protocol(ROOT / args.protocol)
    root = ROOT / (args.root or protocol["result_root"])
    masks_dir = Path(args.masks_dir) if args.masks_dir else ROOT / protocol["masks"]["dir"]

    records, rows = load_runs(root, masks_dir, protocol)
    if not rows:
        raise SystemExit("no complete runs under " + str(root))
    per_run = require_complete_grid(rows, protocol, args.stage)
    per_seed = write_tables(root, records, per_run, protocol)
    branch = pd.read_csv(root / "branch_behaviour.csv")
    clipping = clipping_report(per_run, per_seed)
    atomic_json(root / "clipping_report.json", clipping)

    if args.report:
        print(per_seed.to_string(index=False))
        print("tables written to " + str(root))
        return 0

    isolation = isolation_status()
    verification_path = root / "verification_summary.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file() else {"verified_runs": 0, "runs": []}
    )
    verification_ok = verification.get("verified_runs", 0) == len(per_run)
    entries, manifest_hash = artifact_manifest(records, root)
    decision = decide(per_run, protocol, args.stage)
    checks = {
        "runs": len(per_run),
        "grid_complete": True,
        "all_records_audited_with_strict_identity": True,
        "independent_verification": {
            "verified_runs": verification.get("verified_runs", 0),
            "covers_the_whole_grid": bool(verification_ok),
            "identity_checked": verification.get("identity_checked"),
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
    name, task = STAGE_RESULT[args.stage]
    payload = {
        "task": task,
        "stage": args.stage,
        "protocol_version": protocol["protocol_version"],
        "protocol_revision": protocol.get("protocol_revision", 1),
        "dataset": protocol["dataset"]["path"],
        "candidate": CANDIDATE,
        "controls": list(CONTROLS),
        "outcome": outcome,
        "rules_passed": bool(decision["passed"]),
        "complete": True,
        "decision": decision,
        "clipping": clipping,
        "checks": checks,
        "artifact_manifest": entries,
        "artifact_manifest_sha256": manifest_hash,
        "notes": [
            (
                "thresholds are development decision rules, not ecological "
                "effect sizes and not significance tests"
            ),
            (
                "training seeds test optimisation stability; they are not "
                "independent ecological replicates"
            ),
            "the outer test was never read while this decision was produced",
        ],
    }
    atomic_json(root / (name + ".json"), payload)
    write_decision_page(root, payload, decision, isolation, args.stage, branch,
                        clipping)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("notes", "artifact_manifest", "clipping")},
                     indent=2, ensure_ascii=False))
    print("wrote " + str(root / (name + ".json")))
    print("wrote " + str(root / (name + ".md")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
