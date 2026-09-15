# H3-A v1 冻结协议

日期：2026-09-15。任务来源：`docs/h3_architecture_task_plan_20260915.md`（T02）。
机器可读版本：`configs/h3a_v1.json`。两者必须一致；代码只读 `configs/h3a_v1.json`，
本文件用于解释"为什么这样定"，不引入新的可选项。

## 1. 问题与结构

在 log1p(DOC) 空间计算 `z_hat = base + correction`：

| arm | 结构 | 输入 |
|---|---|---|
| ENV | 两层 64 维 MLP + 标量预测头 | 21 列非 DOC 环境特征 |
| H2X | 现有两层、64 维静态 H2X（`TransportGCNImputer`，`gate_mode=static`，生态编码 32） | 原节点通道、DOC 可见通道、生态编码、边特征 |
| H3A | 与 ENV 相同的基础分支 + 与 H2X 同形的静态修正分支 | 基础分支读 ENV 输入；修正分支读 H2X 输入 |

本轮只回答"这个结构是否比现有 H2X 和独立环境模型更稳定、更准确"。
**不同时**加入动态 gate、多尺度边、历史 DOC 或 GRU。21 列输入见第 3 节。

## 2. 已确定值（不存在"跑完再决定"的项）

| 项 | 冻结值 |
|---|---|
| 数据 | `data/processed/mississippi_graph_v05.pt`（571 站 × 652 月，已应用冻结的协变量质量规则） |
| 优化器 | Adam，lr=0.001，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| dropout | 0.1（基础分支、修正分支、MLP 一致） |
| 轮数 | max_epochs=200，patience=20，min_delta=1e-6 |
| 早停依据 | val_target 上**未截断**的 log1p MSE |
| 损失 | train 单元上的 log1p MSE |
| 更新单元 | 一个月 = 一次全图前向后向；每轮重新打乱月份顺序 |
| 重遮蔽 | 每轮把 train 单元重新随机对半：一半可见作上下文，一半隐藏并承担损失 |
| 梯度裁剪 | 全局范数 1.0 |
| 截断 | log1p 预测截断到 [min(train 对数标签), quantile(train 对数标签, 0.995)] |
| 原始对数输出 | 另存，不参与截断 |
| 训练种子 | pilot：0/1/2；扩展：0/1/2/3/4 |
| 随机源 | `torch.manual_seed(seed)`、`numpy.random.seed(seed)`、单元置换用 `default_rng(seed)` |

### 参数量（T02 记录，实现后由运行记录复核）

- ENV：21→64→64→1，约 5.6k 参数。
- H2X：输入 14 通道 + 生态嵌入 32 = 46 维；两层 `GatedDirectedConv`（edge_dim=6，门宽 16）；**23461** 参数（与 `experiments/dynamic_gate_v1` 的 `static` 臂同构）。
- H3A：ENV 基础分支 + H2X 同形修正分支，参数量为两者之和。

**不使用旧 H2X 最佳分数代替新对照**：H2X 必须在新划分、新训练器下重跑。

## 3. 21 列环境特征视图

固定顺序（`river_graph.models.h3.ENV_FEATURE_NAMES`）：

```
0  temp_std          1  temp_observed     2  flow_std          3  flow_observed
4  season_sin        5  season_cos        6  lat_std           7  lon_std
8..20 regime_0..12   （v04 生态/水文属性 13 列）
```

- 由 `GCNDocModel._build_inputs` 产出的既有标准化张量切片而成，
  **不新写第二套填补/标准化算法**：水温、流量、经纬度用 train 单元的均值/标准差，
  缺失标记原样，季节项用月份正余弦，13 列属性用列均值/标准差。
- 环境视图**不含** DOC 值，也不含 DOC 可见性通道（切掉通道 8/9）。
  改变 DOC 或其可见性不改变环境视图。
- 通道 1/3 承载的是 `x_mask`，其值 1 表示**已观测**；名称使用 `*_observed`，
  以免名称与数值语义相反。
- 视图函数不原地修改输入张量。

## 4. 初始化

- 基础分支（ENV 和 H3A 的基础分支）使用**相同子种子** `seed + 0`，模块结构相同，
  因此两者的初始参数逐元素相同；构造顺序不影响结果（用显式 `torch.Generator` 重抽，
  抽取方案等于 PyTorch `nn.Linear` 默认初始化）。
- H2X 分支用 `seed + 0`，与旧 `GCNDocModel` 在 `manual_seed(seed)` 下的默认初始化逐元素一致，
  所以 H2X 臂确实是"现有静态 H2X"而不是新模型。
- 修正分支用独立子种子 `seed + 200003`。
- 修正分支的最终线性头权重与偏置**严格置零**。

零头初始化会让修正骨干在**第一次**反向时没有梯度（`dL/dh = Wᵀ·grad = 0`）。
这是预期行为：第一头自身和基础分支从第一步起就有梯度，骨干从第二步起获得梯度。
T07 对此有断言，不把它当作缺陷。

基础与修正的数值分解**不是**可识别的生态过程分解。修正分支含 self 通路，
空图时 correction 不一定为零；因此要评估邻居信息贡献必须另训无消息对照（T18）。

## 5. 训练设置

首版不加预训练、不加冻结、不加辅助损失、不加可学习融合系数，联合优化最终输出的 log1p MSE。

可见性协议（`configs/h3a_v1.json → training.visible_roles`）：

可见性由 mask 里的 `visible_roles` 显式声明，不再由“某个键是否存在”隐含决定：

| 阶段 | 可见 | 始终隐藏 |
|---|---|---|
| 训练前向 | context（若声明）+ train 的可见一半 | train 的隐藏一半、val、val_context、test |
| 选择前向 | 仅 `visible_roles` 列出的角色（并集），外加 train | val、test |
| 验证前向 | 与选择前向相同 | val、test |
| 冻结评估前向 | 全部非 test 观测 | test |

`visible_roles` 只允许取自 `{context, train, val_context}`；声明 `val` 或 `test` 会被直接拒绝。
某个角色即使存在但未被声明，也保持关闭，并且**仍然记录在 split 中**，
不会让观测单元无解释地消失。

val 真值**只能**通过评分影响选择；不能进入输入、不能进入损失。

## 6. 验证划分（T03）

外层 test 位置保持不变且始终隐藏。本修订的内部 mask 见
`experiments/h3a_v1r2/masks/` 与 `experiments/h3a_v1r2/split_manifest.json`；
修订 1 的 mask 与结果完整保留在 `experiments/h3a_v1/`。

| mask | 内部 val 形式 | 可见性 | 说明 |
|---|---|---|---|
| `e1_r20_seed42/43/44` | 沿用原 train/val/test | `train` | 单元随机划分场景 |
| `e2a_strict` | 原 strict val 的 80% | `train` | **无未来 DOC context 的严格时间外推对照**；val_context 记录但不开放；沿用原 3184 个 test 目标 |
| `e2b_partial` | 原 val 的 80% | `train,context,val_context` | 运行网络下的未来重建；选择前向开放 val_context |
| `e3_internal_seed42/43/44` | 约 10% 非 test 观测站的全部月份 | `train` | 站点留出场景，保存站点名单与连通分量分布 |
| `e2b_partial_hiddenctx`（诊断，不属于 8 个关键场景） | 与 e2b 完全相同的目标池 | `train,context` | 只改变选择期可见性；**不能替代 E2a** |

修订 1 曾把 `e2a_partial` 当作 E2a 使用，但它继承的是原 E2b 的外层 test 与
未来 context，最终推理仍能看到 637 个未来 DOC，所以它不是"无同期 DOC"的对照。
本次改名为 `e2b_partial_hiddenctx` 并明确标为验证策略诊断，同时恢复真正的
`e2a_strict`。

E2a/E2b 的目标池不同，二者的差异同时包含时间外推场景差异；而在同一 partial
目标池上，选择可见性的差异由 `e2b_partial` 与 `e2b_partial_hiddenctx` 配对给出。

这是新验证协议：旧实验只能作外部参照。

## 7. 指标与汇总

指标定义与冻结基准完全一致（`river_graph.experiments.evaluate.metrics`），
在**内部验证单元**上、mg/L 空间计算：`mae / rmse / r2 / log_mae / log_rmse / log_r2 / pbias / n`。
另外报告 `raw_log_rmse`、`raw_log_mae`（未截断 log1p 输出上的误差，即早停依据）。
headline 指标用截断后的预测，与冻结基准口径一致。

汇总：先在同一场景内对 mask 等权平均，再对训练种子平均；跨 mask 不汇总单元。
单个 seed 的结果只用于检查运行健康度，不用于性能晋级。

pilot / expand 阶段**不导出外层 test 指标**。外层 test 只能由单独的冻结评估入口
（`scripts/run_h3_frozen_eval.py`）导出，且必须同时满足：T17 判定为 promote、
判定文件标为 complete、120 个扩展配置齐全、磁盘上的运行集合与判定记录的清单哈希一致、
每条记录通过严格身份核验。仅 pilot 晋级、扩展缺项或身份不符都会拒绝导出。
河网贡献主张另需 T18 无消息对照，必须用 `--claim network` 显式声明。

## 8. 晋级规则

### 8.1 pilot（T16）

1. E2b 平均验证 `log_rmse` 相对 ENV、H2X **均**降低至少 2%；
2. E1/E3 的平均验证 MAE 相对 H2X 退步均不超过 3%；同时**如实报告与 ENV 的差距**；
3. E2b 至少 2/3 训练种子**同时**优于两个对照；
4. 所有产物审计与标签隔离检查通过。

### 8.2 扩展（T17）

同样的 2% 收益与 3% 退步保护，另加：

5. E2b **至少 4/5** 训练种子同时优于两个对照；
6. E1/E3 先在各自 mask 内计算，再对 mask 与种子等权汇总；E2a 与 E2b 分开报告；
7. 判定必须绑定**确切的运行集合**：逐运行的 config_hash 与 checkpoint 哈希组成的
   清单哈希写入判定文件，冻结评估入口会重新计算并要求完全一致。

只有三类结论：晋级、实现缺陷待修复、当前结构未获支持。
阈值是开发决策标准，**不是**生态效果阈值或统计显著性标准。
未通过时在现有验证结果上诊断，不自动扩大训练、不追加随机种子挑最好结果；
下一项结构变化形成新协议和新目录。

## 9. 预算

| 阶段 | 新增正式配置上限 |
|---|---:|
| 冒烟（T13，3 次短训练，最多 9 个 epoch） | 0 |
| 首组真实实验（T14） | 3 |
| 补齐 pilot（T15） | 24 |
| **T16 里程碑合计** | **27** |
| 修订 2 重新确认（R4：先 E2b 9 次，再补齐 27 次） | 27（与修订 1 的结果分属不同身份） |
| 条件性五种子扩展（T17） | 93（累计 120） |
| 条件性无消息对照（T18） | 40（累计 160） |

T17–T19 按 T16 结果进入，不与第一轮同时堆叠。

## 10. 修订 2（R1–R4）改了什么

| 项 | 修订 1 | 修订 2 |
|---|---|---|
| 数据 | `mississippi_graph_v04.pt` | `mississippi_graph_v05.pt` |
| 水温输入 | 直接采用提供方数值（最高 1310 °C） | 先过`covariate_quality_v1`单位与物理有效性规则；不合格标为缺失 |
| 环境特征名 | `temp_missing/flow_missing`（值为 1 表示已观测） | `temp_observed/flow_observed` |
| E2a | `e2a_partial`（仍含 637 个未来 context） | `e2a_strict`（无未来 context，val_context 记录但不开放） |
| 可见性 | 由键是否存在隐含决定 | 每个 mask 显式声明 `visible_roles` |
| H2X 依赖 | 需要未提交的 `gate_mode` 接口 | 只用已发布的静态接口；非静态门控直接拒绝 |
| 独立核验 | 只检查记录内部自洽 | 用当前源码/协议/数据/mask 重算身份并逐字段比对 |
| 扩展核验 | 按记录写入阶段过滤 | 按阶段应有的 `(arm, seed, mask)` 网格核验，缺项/多项/重复均失败 |
| 外层 test 门禁 | 仅检查 pilot promote | 要求 T17 判定 + 完整 120 配置 + 清单哈希一致 |
| 截断说明 | 写死的结论 | 由逐运行预测动态生成 |
| 结果目录 | `experiments/h3a_v1` | `experiments/h3a_v1r2`（修订 1 结果完整保留） |

修订 1 的 27 次运行与其 T16 判定全部保留，可用 `experiments/h3a_v1` 复议；
修订 2 的判定只能使用修订 2 的运行，两者不混用。

## 11. 已知限制

- v05 数据保留混合站点类型、年代不匹配等限制；开发结果不替代后续核心河流数据和独立划分上的确认。
- 协变量质量规则只覆盖单位与物理有效性；WQP 缓存里仍有 57 个 DOC 标签为 0，
  本轮**未**改动 DOC 标签，留待下一轮单独决定。
- 训练种子只检验优化稳定性，不提供新的独立流域证据。
- 无消息对照的空图使关系参数不活跃，检验的是"邻居消息是否必要"，不等于完美的有效容量匹配。
- 运行身份记录工作树文件的 sha256；本机 `core.autocrlf` 为 true，跨机复算需先固定行尾。
