# 阶段 A 报告：证据盘点、保存闭环、多种子分析与图表校准（2026-09-12）

依据 `docs/execution_plan_20260912.md` 执行。**本轮真实训练次数为 0**：没有执行 `run_freeze.py`，没有模型重训，没有为估时试跑。

## 1. 本轮做了什么

| 阶段 | 交付 | 落地文件 |
|---|---|---|
| A0 证据盘点 | 身份清单、复算核验、冲突记录、70 个历史预测恢复、输入白名单 | `scripts/audit_artifacts.py`、`scripts/recover_historical_predictions.py`、`scripts/build_prediction_manifest.py`、`experiments/analysis/evidence_inventory_20260912.md` |
| A1 保存闭环 | 修 R1；原子写入；provenance sidecar；审计模式；mock 回归 | `src/river_graph/experiments/predictions.py`、`provenance.py`、`scripts/run_gnn.py`、`scripts/run_ladder.py`、`docs/run_gnn_prediction_storage.md` |
| A2 多种子 | 120 次运行清单、两级聚合、配对差值、主表重建 | `scripts/analyze_multiseed.py`、`experiments/analysis/multiseed_report_20260912.md` |
| A3 历史诊断 | 逐站/区域/图角色/极端值/类型/bootstrap 诊断 | `scripts/analyze_predictions.py`、`experiments/analysis/predictions_report_20260912.md` |
| A4 图表报告 | Figure 1 重做、新增 Figure 2、README 校准、本报告与后续清单 | `scripts/figure1_evolution.py`、`scripts/figure2_multiseed_variation.py`、`README.md`、`docs/` |

提交：`bc145c7`（A0）、`2e688b0`（A1）、`7df2748`（A2）、`7db78f1`（A3）、本报告所在提交（A4）。全部推送到 `https://github.com/yangzc2004-bit/river_graph`。

测试与静态检查：`pytest` **65 项通过**（原 43 项 + 新增 22 项），`ruff check .` 无告警。

## 2. A0：证据身份的核验结果

**预测文件身份**（`scripts/audit_artifacts.py --verify`，从 mask 的 test 平坦索引重建测试单元后独立复算）：

| 状态 | 文件数 | 含义 |
|---|---:|---|
| `verified`（满覆盖且与冻结表一致） | 70 | 可直接进入分析 |
| `verified_partial_coverage` | 13 | 指标一致但覆盖不足（全部为 B1_kriging） |
| `zero_coverage` | 1 | `B1_kriging × e2a_strict`，test 预测全为 NaN（n=0，设计行为） |
| `conflict` | 1 | `G0_gcn_none__e1_r20_seed42.parquet` |

**冲突的完整结论**（`evidence_inventory_20260912.md` §4）：

- 该文件在 `256d08e` 入库（blob `05691dd`），被 `2b67bf0` 改写为另一个模型运行的结果（blob `7f4f22c`），同一提交还删除了 69 个预测。
- 从 `256d08e` 恢复的版本复算为 MAE **1.822689** / R² **0.169039**，与冻结表**逐位一致**；当前工作区版本为 1.831044 / 0.176242。
- 因此**冻结表与历史批次自洽**，冲突文件属「同名不同运行」，本轮排除，冻结表保持权威。

**历史恢复**：`scripts/recover_historical_predictions.py --verify` 恢复 **70** 个文件（69 个被删 + 1 个被覆盖），复算 **70/70 与冻结表一致**（最大 |ΔMAE| = 2.1e-07），测试单元不匹配数 **0**。恢复目录已被 gitignore（可由 git 对象库重建）。

**结论**：`benchmark.csv` 的 **154 行现在全部有来源明确、指标可复算的逐样本预测**；此前 `experiments/predictions/` 只有 85 个。

## 3. A1：R1 的修复与不训练的证明

原缺陷：指标 JSON 命中即 `cached, skip`，即使传了 `--save-predictions` 且 parquet 缺失也不报告。实测暴露面 **205** 个 (模型, mask) 组合。

修复后的行为（`CacheState`：`absent / metrics_only / predictions_only / complete`）：

| 状态 | 传 `--save-predictions` | 传 `--rebuild-predictions` | 都不传 |
|---|---|---|---|
| `metrics_only` | 报缺失，末尾退出码 **2**，**不训练** | 显式重跑该 mask 一次，指标与预测同源 | 复用指标 |
| `complete` | `cached, skip`，不训练 | 同上（除非 `--force`） | 同左 |
| `predictions_only` | 从预测的 test 行复算指标，不训练 | 同左 | 同左 |

其他保障：

- **原子写入**：临时文件 + `fsync` + `os.replace`；中断残留的 `.tmp` 不会被识别为完整缓存。
- **provenance sidecar**：`config_hash`（种子/架构/variant/lr/wd/dropout/env 设置/模型名）+ 数据集与 mask 的**内容 sha256**；配置不同则拒绝覆盖，需 `--force`。
- **文件名保持 `{model}__{mask}.parquet` 不变**，因此既有工具链（`recompute_metrics.py`、`run_freeze.py`）无需改动。
- `scripts/run_ladder.py` 作为未来正式训练的包装入口，强制携带保存开关，并提供不加载数据的 `--dry-run`。
- `scripts/recompute_metrics.py` 现在先报告「重写会丢 69 行、改 1 行」并**默认拒绝**覆盖冻结表。

**不训练的证明**：`tests/test_run_gnn_cache.py` 用计数 mock 模型工厂断言构造次数——新运行每 mask 训练一次、完整缓存构造 **0** 次、`metrics_only` **不自动重训**、`--rebuild-predictions` 才训练、身份不符被拒绝、中断 `.tmp` 不算完整缓存。共 14 项测试。

## 4. A2：多种子主结论

输入 15 个 JSON，校验 **120 次运行**（3 模型 × 5 训练种子 × 8 key mask）全部齐备。重建的 `benchmark_multiseed.csv` 与原文件数值一致（`mae_std` 仅差 1.0e-16）。

**可保留的结论**：

1. H2X 在 8 个 key mask 的平均 MAE 上优于 H1/H2；E2a、E2b、E3 的配对差值**每一对都为负**（5/5、5/5、15/15）。
2. `e2b_partial` 平均 R² = **0.447**（5 训练种子，sd 0.033）。
3. E2a/E2b 上 H2X 训练波动明显小于 H2（MAE sd 0.039/0.026 对 0.152/0.151）。

**必须同时说明的边界**：

- **E1 不能叙述为 H2X 提高了 R²**：H2 = 0.439613、H2X = 0.437845，配对差值均值 **−0.0018**。
- H2 相对 H1 的优势**只在 E1 稳定**（15/15）；E2a/E2b 配对均值 ≈0 且区间宽，**不支持**「transport 门控改善未来重建」。
- 15 次训练不是 15 个独立空间样本；训练种子离散度是训练随机性，不是逐样本预测不确定性。
- E1/E3 为 3 mask 等权平均；平均 R² ≠ 合并样本重算 R²。

## 5. A3：历史诊断的主结论

- **误差高度集中于极端值**：154 次预测中，真实 DOC 最高 1% 的单元贡献平方误差的**中位数 74.2%**；E3 seed43 为 **95.6%**。因此只报 RMSE 的模型比较会被极少数单元主导。
- **区域**：HUC2 直接取自 `graph_nodes.huc_cd`（不由站号前缀推断）。E3 下 HUC2=10（Missouri）逐单元加权 MAE 最高（H2X 2.656，106 站）；HUC2=8（仅 4 个 test 站）**顺序反转**（基线 0.610 最好、H2X 2.180 最差），说明 H2X 的优势不是全区域一致。
- **图角色**：无上游监测邻居的站点在所有场景误差更高，但站点均值基线上顺序相同 → 主要反映站点难度，不是图模型特有缺陷。
- **站点类型**：ST 站点误差高于其他类型，**包括基线模型**（B0 E1：1.59 对 0.82）→ 描述性敏感性，**不能**冒充 ST-only 验证。
- **站点 06438000**：观测统计由数据集逐月标签直接计算——28 个观测月（1978-03 → 1988-09），中位数 6.75、均值 58.22、**最大 460 mg/L**；所有模型严重低估高值月；**成因未定**，不作因果主张。
- **置信区间**：按**站点聚类**重采样（整簇进出），纠正了此前按行抽样等于抽「站点 × 划分」组合的问题。

## 6. A4：图表校准

**Figure 1（重做）**

- 修好了 R2 标签碰撞：原脚本给所有标签统一上移 8 点，G0/E3 标签压到 E2b 标记上。新版本改为**每个场景一个面板**（单系列、无碰撞可能），并加入**自动标签避让**（逐点上移直到像素框不重叠）。
- 新增**布局校验**：`check_layout()` 在图渲染后检查文字是否重叠、值标签是否越出面板、图注是否被裁切；不通过就报错退出。校验过程中确实捕获了两个真实缺陷（标签越界、`sharey` 导致 E1 数据被裁到轴范围之外）。
- 保留已核对的 12 个均值不变；图注写明 G0 历史来源、H 系列 5 训练种子、E1/E3 三 mask 等权平均、无误差条的原因，以及「H2X 同时增加生态信息与编码器，不能归因给编码器」。
- 导出 PNG + **SVG + PDF** + source-data CSV（逐 mask 与聚合两级）。

**Figure 2（新增）**：使用 A2 得到的**训练种子**标准差（不是旧的 mask 间 `std()`），误差条含义写进标题与图注；G0 因无种子离散度而不出现。

**README 校准**：删除「多种子仍在运行」「H2 在 E2 最优」「每级只改一件事所以可归因」等旧叙述；历史参考表与新五种子主表**分表标注**；r40/r60 明确标为历史批次；新增「证据状态与已知限制」一节。

## 7. 本轮明确未做（不要把完成阶段报告当成研究闭环）

- 未修 `env_groups` 取列错误（B0）；未建立 ST 核心版本与重新构图（B1）；未做同协议 H2/H2E/H2X 与有效无图对照（B2）；未做 r40/r60 多种子、空间分块与外流域评估（B3）；未做 H2X + 历史输入基线（B4）。
- 未生成 H1/H2/H2X 五训练种子的逐样本预测（需 120 次训练），因此**没有**五种子集成、残差地图、逐站区间。
- 未做数据清理与 mask 改动。
- **未更新 `river_graph.pdf`**：其构建脚本 `_pdf_work/rebuild_river_graph.py` 被 gitignore，且当前 Python 环境**未安装 reportlab**，无法在本环境重建。PDF 与 README 目前**不同源**，这是本轮未闭合项（见 `followup_experiments_20260912.md`）。

## 8. 验收对照

| 计划验收项 | 结果 |
|---|---|
| 能区分历史预测、新多种子指标、冲突文件、缺失文件 | 达成（A0 清单 + manifest + 冲突记录） |
| 不执行真实训练即可证明保存/缓存行为正确 | 达成（14 项 mock 回归，训练计数断言） |
| 主表与误差条明确聚合层级 | 达成（`*_sd_seeds` / `*_sd_of_mask_means` 命名，报告写明） |
| Figure 1 的 E1 R² 不叙述为 H2X 提高 | 达成（配对差值 −0.0018，图注与 README 均写明） |
| 脚本、源数据、图表、正文一致 | 达成（图由脚本生成，source-data CSV 随图导出，README 数字来自同一批文件） |
| 缺失问题被明确列出 | 达成（A2 §6、A3 §8、README「已知限制」） |
