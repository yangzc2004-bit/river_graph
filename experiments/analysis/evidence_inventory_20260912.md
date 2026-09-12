# 证据身份与可用性清单（A0，2026-09-12）

依据 `docs/execution_plan_20260912.md` 的 A0 阶段。本轮**没有执行任何真实训练**，也没有改动数据集、mask、现有预测文件或 `benchmark.csv`。本文档只回答「哪些产物可以被本轮分析使用、来源是什么、哪些不能」。

## 1. 本轮基准

| 项目 | 值 |
|---|---|
| 分析基准提交 | `cea452c`（= `origin/main`） |
| 工作区差异 | 仅 `docs/execution_plan_20260912.md` 未跟踪；本轮新增脚本与报告 |
| 数据集 | `data/processed/mississippi_graph_v02.pt`、`v03.pt`、`v04.pt` |
| 数据集规模（实测） | 571 节点 × 652 月；33,048 个观测单元（覆盖率 8.877%）；570 个节点至少有一个观测单元；562 条有向边 |
| mask | `experiments/masks/` 14 个 `.npz` |
| 冻结表 | `benchmark.csv` 154 行（11 模型 × 14 mask） |
| 多种子表 | `benchmark_multiseed.csv` 24 行（3 模型 × 8 key mask 的单次训练指标） |

机器可读清单：`experiments/frozen_results/prediction_manifest_20260912.json`（本轮分析输入冻结清单）。逐文件明细：`experiments/analysis/artifact_verification_20260912.csv`、`experiments/analysis/artifact_inventory_20260912.json`。

## 2. 清单是怎么产生的

- `scripts/audit_artifacts.py --verify`：统计**实际磁盘文件**（不只依赖 `git status`），对每个 parquet 记录模型、mask、数据集、行数、split 分布、`sha256`，并**从 mask 的 test 平坦索引重建测试单元**，再按 `(station, month)` 匹配 parquet 行，独立复算 MAE / R² / RMSE / n 与覆盖率。
- 判定口径：只有「复算指标与冻结表一致（容差 1e-4）」才算 `verified`。覆盖率不足但指标一致的单独记为 `verified_partial_coverage`，**不**与冲突混为一谈。
- `scripts/recover_historical_predictions.py`：从明确提交恢复历史副本并复算核验。
- `scripts/build_prediction_manifest.py`：生成本轮输入白名单。

**本轮无法验证的部分**（明确标注，不用推断补齐）：无法确认历史 parquet 与 `benchmark.csv` 来自哪一次具体训练进程；无完整配置/数据哈希与权重管理；无 CUDA/硬件信息记录；文件时间戳不足以证明批次归属。因此「来源清楚」的边界是：**文件身份、指标一致性、与 mask 的对应关系**三层，而不是「可完整复现的训练闭环」。

## 3. 预测可用性矩阵（按模型 × mask）

`current` = `experiments/predictions/`；`recovered` = `experiments/predictions_historical/`（从 `256d08e` 恢复）。

| 模型 | 冻结表行 | current 预测 | recovered 预测 | 合计 | 本轮可用性 |
|---|---:|---:|---:|---:|---|
| B0_station_mean | 14 | 14 | 0 | 14 | 可用 |
| B1_kriging | 14 | 14 | 0 | 14 | 可用（覆盖率需并列报告，见 §6） |
| B2_random_forest | 14 | 14 | 0 | 14 | 可用 |
| B3_mlp | 14 | 14 | 0 | 14 | 可用 |
| G0_gcn_none | 14 | 1 | 14 | 15 | 可用；seed42 的 current 副本为**冲突文件**，不使用（见 §4） |
| G0_gcn_random | 14 | 0 | 14 | 14 | 可用（仅历史批次） |
| G0_gcn_river | 14 | 0 | 14 | 14 | 可用（仅历史批次） |
| H1_directed_river | 14 | 0 | 14 | 14 | 可用（仅历史批次） |
| H2_transport_river | 14 | 0 | 14 | 14 | 可用（仅历史批次） |
| H2E_transport_river | 14 | 14 | 0 | 14 | 可用（历史 v04 运行） |
| H2X_transport_enc_river | 14 | 14 | 0 | 14 | 可用（历史 v04 运行） |
| **合计** | **154** | **85** | **70** | **155** | 154 个冻结行全部有可核验预测 |

结论：**154 行冻结表现在全部有来源明确的逐样本预测**。此前 `experiments/predictions/` 只有 85 个（缺 69 个），缺的 69 个来自 `2b67bf0` 的删除，已从 `256d08e` 原样恢复。

每个 parquet 的身份信息（`model / mask / dataset_version / 行数 / split 分布 / sha256`）在 `artifact_inventory_20260912.json` 的 `predictions` 字段中，可直接追踪到分析用途。

## 4. 冲突记录：`G0_gcn_none__e1_r20_seed42.parquet`

| 指标 | 当前副本复算 | 冻结表 `benchmark.csv` | 差异 |
|---|---:|---:|---:|
| MAE | 1.831044 | 1.822689 | +0.008356 |
| R² | 0.176242 | 0.169039 | +0.007203 |
| n | 6610 | 6610 | 0 |

**已查明的根因**（不改写历史、只记录）：

1. 该文件在 `256d08e`（"Phase 0 complete"）首次入库，blob `05691dd`，299,067 字节。
2. `2b67bf0` 把它改写为 blob `7f4f22c`，300,986 字节——**这是一次不同的模型运行结果**，不是格式变化。
3. 同一个 `2b67bf0` 还**删除了 69 个** parquet（`git show --name-status` 计数 69 个 `D`）。
4. 冻结表 `benchmark.csv` 中该行从 `256d08e` 起就是 1.822689 / 0.169039，`d3f0655`（"recompute from parquet"）只增加了 PBIAS 列，**没有改变该行的 MAE/R²**。
5. 决定性证据：从 `256d08e` 恢复的该文件版本复算结果为 MAE **1.822689** / R² **0.169039**，与冻结表**逐位一致**（差值 ~4e-08 浮点噪声）。

因此：**冻结表与历史批次自洽，当前工作区的这个文件属于被覆盖的、不在冻结表内的另一次运行。**

**处置决定**

- **保留**当前冲突文件，不"修回"、不覆盖用户文件。
- 本轮分析中该文件**不进入任何汇总**（manifest 中 `used_for_phase_a: false`）。
- 历史 G0 值以 `benchmark.csv` 为权威，复算值存档于 `historical_verification_20260912.csv`。
- 需要 G0 逐单元诊断时，使用恢复副本 `predictions_historical/G0_gcn_none__e1_r20_seed42.parquet`（已核验一致）。
- 该冲突不影响 Figure 1：图使用冻结表 G0 river 行，其对应预测在恢复批次中已核验。

## 5. 历史预测恢复（69 + 1）

- 来源提交：`256d08e`（"Phase 0 complete: benchmark frozen (11 models x 14 masks, 154 parquet predictions)"）。
- 恢复数量：**70** 个（69 个被删除 + 1 个被覆盖的 G0 副本），全部为**历史单次训练批次**。
- 核验结果：70/70 与冻结表一致；`max |ΔMAE| = 2.1e-07`，`max |ΔR²| = 1.4e-07`；测试单元不匹配数 0；覆盖率不足的 0 个。
- 恢复文件目录 `experiments/predictions_historical/` 已加入 `.gitignore`：它是**可由 git 对象库重建的缓存**，不入库，以保证 `experiments/predictions/` 仍是唯一的规范批次。记录见 `experiments/analysis/historical_recovery_manifest_20260912.json`。
- **恢复的是历史结果，不是新五种子预测。** 二者在标题、表名和图注中必须分开。

## 6. 覆盖率：Kriging 系列必须与 n 并列报告

只有 `B1_kriging` 存在非满覆盖，其余模型覆盖率均为 1.0。

| mask | test 单元 | Kriging 实际预测 | 覆盖率 |
|---|---:|---:|---:|
| e1_r20_seed42 | 6610 | 6594 | 99.76% |
| e1_r20_seed43 | 6610 | 6589 | 99.68% |
| e1_r20_seed44 | 6610 | 6591 | 99.71% |
| e1_r40_seed42 | 13219 | 13135 | 99.36% |
| e1_r40_seed43 | 13219 | 13144 | 99.43% |
| e1_r40_seed44 | 13219 | 13156 | 99.52% |
| e1_r60_seed42 | 19829 | 19500 | 98.34% |
| e1_r60_seed43 | 19829 | 19543 | 98.56% |
| e1_r60_seed44 | 19829 | 19441 | 98.04% |
| e2a_strict | 3184 | 0 | **0%** |
| e2b_partial | 2547 | 2422 | 95.09% |
| e3_spatial_seed42 | 5068 | 5050 | 99.64% |
| e3_spatial_seed43 | 4004 | 4000 | 99.90% |
| e3_spatial_seed44 | 4909 | 4899 | 99.80% |

要点：

- `B1_kriging × e2a_strict` 的 test 预测**全为 NaN**，冻结表该行 MAE/R² 为 NaN、`n=0`。这是设计行为（该月无可用于克里金插值的同月观测），**不是**缺失文件，也**不补算**。
- 由于有效样本范围不同，Kriging 的误差量级**不能**直接与其他满覆盖模型逐位比较；A3 中必须同时给出 n 与覆盖率。

## 7. 「有指标、无预测」的缺口：205 个组合

扫描 `experiments/results/*.json` 的每个模型：指标 JSON 中存在、但 `experiments/predictions/` 中无对应 parquet 的 (模型, mask) 组合共 **205** 个。这是 `run_gnn.py` 缓存命中直接 `cached, skip`（计划文档中的 R1）的直接后果，也是 A1 要修的目标。

主要来源（前 5 类）：

| 模型 | 指标数 | 预测数 | 缺失 |
|---|---:|---:|---:|
| G0_gcn_none | 12 | 1 | 11 |
| G0_gcn_random | 12 | 0 | 12 |
| G0_gcn_river | 14 | 0 | 14 |
| H1_directed_river / H2_transport_river | 14 | 0 | 14 / 14 |
| H1f/H2f/H2Xf + `_s1`–`_s4`（15 个文件） | 8 | 0 | 各 8，共 **120** |
| H2X_{hydro,landcover,climate,soil_topo}_river | 4 | 0 | 各 4，共 16 |
| H15_directed_river | 4 | 0 | 4 |

其中 120 个缺失正是「新五种子 × 8 key mask」的逐样本预测——**它们不存在**，见 §8。

## 8. 为什么五种子集成/残差图/逐站区间不能做

- H1/H2/H2X 的 5 个训练种子（`H1f/H2f/H2Xf` + `_s1`–`_s4`）各有 **0** 个 parquet。只有指标 JSON（8 个 key mask）。
- 因此本轮**没有**「五训练种子集成」「五训练种子残差地图」「五训练种子逐站区间」。这些比较在 A2/A3 中显式列为**无法开展**，不用历史单次批次顶替。
- 能在无训练条件下算出的，是**已有 R²/RMSE/MAE 的指标层聚合与训练波动**（A2）；PBIAS 等字段在多种子 JSON 中存在，但 `benchmark_multiseed.csv` 未落盘这些列，缺字段明确标缺失，**不补零、不从另一批次拼接**。

## 9. 本轮冻结的分析输入

白名单见 `experiments/frozen_results/prediction_manifest_20260912.json`：

- **可用（current，83 个）**：`experiments/predictions/` 中除 1 个冲突文件与 1 个零覆盖文件外的全部；零覆盖文件列入 `zero_coverage` 并附带「报告 n/覆盖率、不比较误差量级」的使用约束。
- **可用（recovered，70 个）**：`experiments/predictions_historical/`，仅用于历史 G0/H1/H2 的逐单元诊断，标注来自历史批次。
- **排除**：
  - `G0_gcn_none__e1_r20_seed42.parquet`（current 副本，冲突）；
  - `H2X_{hydro,landcover,climate,soil_topo}_river` 的 16 行指标——受 `env_groups` 取列错误影响（`src/river_graph/models/gcn.py`），**机制结论失效，仅保留历史描述**；默认全特征 H2X 路径不受该处影响。
  - 所有训练种子 1–4 的逐样本预测：不存在。

## 10. 验收自检

- [x] 记录了当前提交、工作区差异；统计了**实际磁盘文件**（85 个 current + 70 个 recovered），没有只依赖 `git status`。
- [x] 每个可用预测都能给出模型、数据版本、mask、指标/预测路径与 `sha256`（`artifact_inventory_20260912.json`）。
- [x] G0 none 冲突文件单列，并在独立目录读取/恢复了明确 Git 提交中的历史副本。
- [x] 对恢复文件复算了 test 指标，并核验了站点、月份、标签与 mask 的对应关系（70/70 一致，0 个单元不匹配）。
- [x] 对含 NaN 的输出（Kriging）同时报告 n 与覆盖率。
- [x] 冻结了本轮分析输入清单；来源未知的预测不进入主分析。

**可用性边界**：能明确区分历史预测（recovered 批）、新多种子指标（仅 JSON，无逐样本预测）、冲突文件、缺失文件；每项分析都能追踪到来源。本轮**未**完成数据清理、机制对照实验与跨区域泛化验证——这些属 B 阶段。
