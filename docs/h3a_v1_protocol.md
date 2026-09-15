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
| 数据 | `data/processed/mississippi_graph_v04.pt`（571 站 × 652 月） |
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
- H2X：输入 14 通道 + 生态嵌入 32 = 46 维；两层 `GatedDirectedConv`（edge_dim=6，门宽 16）；约 3.1 万参数（与 `experiments/dynamic_gate_v1` 的 `static` 臂同构）。
- H3A：ENV 基础分支 + H2X 同形修正分支，参数量为两者之和。

**不使用旧 H2X 最佳分数代替新对照**：H2X 必须在新划分、新训练器下重跑。

## 3. 21 列环境特征视图

固定顺序（`river_graph.models.h3.ENV_FEATURE_NAMES`）：

```
0  temp_std          1  temp_missing      2  flow_std          3  flow_missing
4  season_sin        5  season_cos        6  lat_std           7  lon_std
8..20 regime_0..12   （v04 生态/水文属性 13 列）
```

- 由 `GCNDocModel._build_inputs` 产出的既有标准化张量切片而成，
  **不新写第二套填补/标准化算法**：水温、流量、经纬度用 train 单元的均值/标准差，
  缺失标记原样，季节项用月份正余弦，13 列属性用列均值/标准差。
- 环境视图**不含** DOC 值，也不含 DOC 可见性通道（切掉通道 8/9）。
  改变 DOC 或其可见性不改变环境视图。
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

| 阶段 | 可见 | 始终隐藏 |
|---|---|---|
| 训练前向 | context + train 的可见一半 | train 的隐藏一半、val、val_context、test |
| 选择前向（E2b） | context + train + val_context | val、test |
| 选择前向（其它） | context + train | val、val_context、test |
| 验证前向（E2b） | context + train + val_context | val、test |
| 验证前向（其它） | context + train | val、val_context、test |

val 真值**只能**通过评分影响选择；不能进入输入、不能进入损失。

## 6. 验证划分（T03）

外层 test 位置保持不变且始终隐藏。新内部 mask 见 `experiments/h3a_v1/masks/`
与 `experiments/h3a_v1/split_manifest.json`。

| mask | 内部 val 形式 | 说明 |
|---|---|---|
| `e1_r20_seed42/43/44` | 沿用原 train/val/test | 单元随机划分场景 |
| `e2a_partial` | 原 val 的 80% `val_target` | 原 val 的 20% 归 val_context，但选择时**全部隐藏** |
| `e2b_partial` | 同一 val_target | 选择前向开放 val_context |
| `e3_internal_seed42/43/44` | 约 10% 非 test 观测站的全部月份 | 站点留出场景，保存站点名单与连通分量分布 |

E2a/E2b 共用同一 val_target，但选择条件不同，**可以得到不同的最佳权重**——
不沿用旧实验"权重必然相同"的假设。

这是新验证协议：旧实验只能作外部参照。

## 7. 指标与汇总

指标定义与冻结基准完全一致（`river_graph.experiments.evaluate.metrics`），
在**内部验证单元**上、mg/L 空间计算：`mae / rmse / r2 / log_mae / log_rmse / log_r2 / pbias / n`。
另外报告 `raw_log_rmse`、`raw_log_mae`（未截断 log1p 输出上的误差，即早停依据）。
headline 指标用截断后的预测，与冻结基准口径一致。

汇总：先在同一场景内对 mask 等权平均，再对训练种子平均；跨 mask 不汇总单元。
单个 seed 的结果只用于检查运行健康度，不用于性能晋级。

pilot / expand 阶段**不导出外层 test 指标**。外层 test 只能由单独的冻结评估入口导出（T19）。

## 8. 晋级规则（T16）

1. E2b 平均验证 `log_rmse` 相对 ENV、H2X **均**降低至少 2%；
2. E1/E3 的平均验证 MAE 相对 H2X 退步均不超过 3%；同时**如实报告与 ENV 的差距**；
3. E2b 至少 2/3 训练种子**同时**优于两个对照；
4. 所有产物审计与标签隔离检查通过。

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
| **本轮合计（T16 里程碑）** | **27** |
| 条件性五种子扩展（T17） | 93（累计 120） |
| 条件性无消息对照（T18） | 40（累计 160） |

T17–T19 按 T16 结果进入，不与第一轮同时堆叠。

## 10. 已知限制

- v04 数据保留混合站点类型、年代不匹配等限制；开发结果不替代后续核心河流数据和独立划分上的确认。
- 训练种子只检验优化稳定性，不提供新的独立流域证据。
- 无消息对照的空图使关系参数不活跃，检验的是"邻居消息是否必要"，不等于完美的有效容量匹配。
