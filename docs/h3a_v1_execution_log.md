# H3-A v1 执行日志（T01–T16）

日期：2026-09-15。计划来源：`docs/h3_architecture_task_plan_20260915.md`。
冻结协议：`configs/h3a_v1.json` + `docs/h3a_v1_protocol.md`。

按计划的执行规则，每个任务记录完成状态、文件、验证证据和剩余问题。
本轮完成计划定义的**首个完整里程碑 T16**；T17–T19 为条件性扩展，进入条件已满足，
但尚未执行（见文末）。

## 总览

| 任务 | 状态 | 主要文件 | 验证证据 |
|---|---|---|---|
| T01 建立实施基线 | 完成 | `scripts/h3_preflight.py`, `experiments/h3a_v1/preflight.json` | Git HEAD `90794e2`；16 个已跟踪文件有用户改动、1979 个未跟踪路径已逐条记录；改动前 pytest 基线 210 passed |
| T02 冻结结构与开发协议 | 完成 | `configs/h3a_v1.json`, `docs/h3a_v1_protocol.md` | 所有架构/训练/早停/截断/种子/指标/晋级项均有确定值；参数量 ENV 5633、H2X 23461、H3A 29094（H3A = ENV + H2X 同形修正分支） |
| T03 场景匹配的内部验证划分 | 完成 | `src/river_graph/experiments/h3_masks.py`, `scripts/build_h3_masks.py`, `experiments/h3a_v1/masks/`, `experiments/h3a_v1/split_manifest.json` | 8 个 mask；外层 test 与冻结 mask 逐元素相同；角色互不重叠；E2a/E2b 共用同一 val_target；E3 验证站无 train 标签；重建可逐字节复现 |
| T04 统一环境特征视图 | 完成 | `src/river_graph/models/h3.py` | 21 列逐元素核对（独立期望值）；改变 DOC 值/可见性不改变视图；不原地修改输入 |
| T05 实现 ENV 基础模型 | 完成 | `src/river_graph/models/h3.py` | 输出形状、孤立节点、与其他节点/边/DOC 无关；宽度不符时报错 |
| T06 实现 H3A 加法输出结构 | 完成 | `src/river_graph/models/h3.py` | eval 下初始 `total == 同初始化 ENV` 且逐元素相等；`total == base + correction`；只在对数尺度相加 |
| T07 可学习性与边界行为 | 完成 | `tests/test_h3.py`（19 项） | 首次反向后骨干无梯度、第二次起非零；修正头能学正/负修正；空边无 NaN 且 self 通路仍在；邻居信息可改变图分支输出 |
| T08 共同训练器与场景验证器 | 完成 | `src/river_graph/experiments/h3_training.py` | 三臂共用同一逐月更新、重遮蔽、可见性与早停逻辑；每轮训练损失、验证损失、最佳轮数、实际轮数均落盘 |
| T09 隐藏标签隔离 | 完成 | `tests/test_h3_training.py`（32 项） | 改 test 真值不改变训练/选择/输出；val 真值只影响选择；val_context 仅在 E2b 前向可见；ENV/H2X/H3A 三臂均测 |
| T10 运行身份与产物保存 | 完成 | `src/river_graph/experiments/h3_runs.py` | 身份不同拒绝覆盖；缺权重或预测无法标 complete；复用前重新审计；pilot 记录 `test_metrics` 恒为 `null` |
| T11 分支结果与独立复算 | 完成 | `scripts/verify_h3.py`, `tests/test_h3_runs.py`（14 项） | 重载最佳权重独立重建预测，与保存值在 1e-6 内一致；总指标可复算；H3A 的 base-only 输出与独立训练 ENV 不相同 |
| T12 任务清单与运行入口 | 完成 | `scripts/run_h3.py`, `scripts/run_h3_frozen_eval.py` | `--dry-run` 列出 27 条待跑配置且不训练；`--verify-only` 不训练；中断后续跑按身份复用 |
| T13 三组冒烟测试 | 完成 | `experiments/h3a_smoke_v1/`, `scripts/h3_smoke_report.py` | 3 条链路（150 站子集、各 3 epoch）全部保存/恢复/审计通过；无 NaN、无隐藏标签泄漏；旧静态前向兼容测试 93 passed |
| T14 第一组真实实验 | 完成 | `experiments/h3a_v1/runs/` 等 | ENV/H2X/H3A × E2b × seed 0 共 3 次，身份与预测核验通过 |
| T15 补齐三种子 pilot | 完成 | `experiments/h3a_v1/*.csv` | 27 次全部完成并逐条独立复算（`verification_summary.json`） |
| T16 架构晋级决定 | 完成 | `experiments/h3a_v1/pilot_decision.json`, `pilot_decision.md` | 四类冻结规则全部通过，结论 **晋级**（措辞应为"达到 pilot 扩展标准"） |

## T16 结果摘要

27 次正式配置（3 组 × 3 个开发 mask × 训练种子 0/1/2），共 13455 秒 CPU≈3.74 小时。

| 规则 | 阈值 | 实测 | 通过 |
|---|---|---|---|
| E2b 平均验证 log_RMSE 相对 ENV | ≤ −2% | −7.21% | 是 |
| E2b 平均验证 log_RMSE 相对 H2X | ≤ −2% | −5.69% | 是 |
| E1 平均验证 MAE 相对 H2X | ≤ +3% | −0.23% | 是 |
| E3 平均验证 MAE 相对 H2X | ≤ +3% | −1.92% | 是 |
| E2b 同时优于两对照的训练种子 | ≥ 2 | 2/3（seed 0、2） | 是 |
| 产物审计与标签隔离 | 全部通过 | 27 条审计 + 76 项测试 | 是 |

同时如实报告与 ENV 的差距：E1 MAE −8.21%、E3 MAE −3.96%。
未截断（原始 log1p）输出上 H3A 仍优于两对照，但 E1 相对 H2X 仅 −0.05%，基本持平。

27 次运行全部触发早停（30–123 epoch）。这只说明 200 轮上限未触及，
**不能**说明 patience=20 没有约束训练：每次都在 best_epoch + patience 处停止，
正是 patience 在起作用。修订 2 的 `training_trajectory.csv` 增加了
`stopped_at_patience` 列来直接区分这两种情况。

> 措辞更正（修订 2）：T16 的结论应表述为"**达到 pilot 扩展标准**"，
> 而不是"结构收益成立"。后者容易被读成已完成泛化确认。
> `pilot_decision.json` 与 `pilot_decision.md` 是冻结的历史记录，未作改写；
> 本行即为更正说明。

## 剩余问题

1. **E2b 的头条指标使用截断后的预测**（与冻结基准口径一致）。独立训练的 ENV 在
   E2b 上有极少数单元越界，截断显著改善其分数：ENV 原始 log_RMSE 0.3696 对
   截断后 0.2999。因此 E2b 相对 ENV 的 −7.21% 有一部分来自截断效应；相对 H2X 的
   −5.69% 不受此影响（H2X 截断比例 0.06%）。原始输出上的对比已一并报告。
2. **E1 上 H3A 与 H2X 基本持平**（log_RMSE −0.04%，MAE −0.23%），E3 上小幅领先
   （−2.44% / −1.92%）。收益主要集中在 E2b 的未来重建场景。
3. **base/correction 的数值分工不是可识别的生态过程分解**。H3A 的 base 幅度
   （≈0.67）小于独立训练的 ENV（≈1.56），因为目标由两个分支共同承担。
4. **本轮没有读取外层 test**。外层 test 只能由 `scripts/run_h3_frozen_eval.py`
   在 `pilot_decision.json` 判定为 `promote` 后导出，且只能当作开发基准。
5. **v04 数据的限制仍然适用**：混合站点类型、年代不匹配；训练种子只检验优化
   稳定性，不提供新的独立流域证据。
6. **未提交的用户改动保持原样**：T01 记录的 16 个已跟踪改动文件与全部未跟踪
   路径均未被本次工作修改、还原或提交。
7. **行尾与身份哈希**：运行身份记录的是工作树文件的 sha256。本机 `core.autocrlf`
   为 true，因此在 Windows 上做全新克隆会把源码检出为 CRLF，进而改变源码哈希，
   使 `verify_h3.py` 的源码身份校验不通过（预测与权重的校验不受影响）。
   在本仓库工作副本内复算不受影响；跨机复算需先固定行尾。

## T17–T19（未执行）

T16 判定为晋级，因此 T17 的进入条件已满足，但本轮按计划的"分阶段"要求停在 T16。

- **T17**：三组 × 8 key masks × 5 seeds = 120 次，其中 27 次 pilot 按身份复用，
  新增 93 次。按本轮实测（h3a 平均 733 秒/次、h2x 558 秒/次、env 204 秒/次），
  新增计算量约 14–15 小时 CPU；本机 8 物理核并行约需 2.5–3.5 小时墙钟。
- **T18**：无消息对照，最多新增 40 次（约 6 小时 CPU），进入条件是 T17 的结构收益成立。
- **T19**：冻结评估入口已实现并上锁（`scripts/run_h3_frozen_eval.py`），
  导出外层 test 前需要 T17（河网收益主张还需 T18）。

本轮不启动 T17–T19，也不据此调整任何阈值。

> **修订 2（2026-09-15）**：review 发现输入质量、可复现依赖与后续阶段控制问题后，
> 已执行 R1–R4 修复，并在新数据版本与新目录下重新确认 pilot。
> 详见 `docs/h3a_v1r2_repair_log.md`。
> 因此"复用原 27 次、只加 93 次"的承诺**不再成立**：数据身份已改变，
> 且旧输入参与过模型选择，旧结果只能作为修订 1 的历史记录保留。
