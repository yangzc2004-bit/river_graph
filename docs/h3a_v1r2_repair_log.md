# H3-A v1 修复阶段记录（R1–R4）

日期：2026-09-15。来源：\`docs/h3a_v1_review_20260915.md\`。
对象：提交 e837f1b/ce261d0 的 T01–T16 实现与 27 次 pilot。

本轮只做审查要求的修复与在新身份下的重新确认，不扩大训练规模，
不改变任何冻结阈值，不读取外层 test。

## R1：输入质量审计与新数据版本

### 定位

站点 05357225，2017-06-26：提供方缓存记录
\`Temperature, water\` = **1310.0 deg C**，单位字段为 \`deg C\`，
\`Result_MeasureStatusIdentifier\` 为 \`Accepted\`，USGS pcode 00010。
该值进入 v04 月度水温通道，标准化后为 147.48，而训练集水温标准化范围是
[-1.57, 2.63]，三组模型的最大未截断误差都落在该站点月份。

### 规则（\`src/river_graph/data/quality.py\`，\`covariate_quality_v1\`）

规则只依据**单位与物理有效性**，不使用模型分数、残差或任何评估标签。

| 变量 | 允许单位 | 物理范围 | 说明 |
|---|---|---|---|
| temperature | \`deg C\` | [-5, 40] °C | 华氏行标为缺失，**不做换算**：该组内存在换算后仍不可能的值（7 月 25.5 °F、10 月 4 °F），单位标签本身不可信 |
| ph | \`standard units\` | [0, 14] | — |
| spec_conductance | \`uS/cm\` | [0, 1e5] | — |
| discharge | 任意（NWIS 已归一） | 保留符号，\|q\| ≤ 3e6 cfs | 负流量来自潮汐/回水河段，是真实观测 |
| doc | \`mg/L\`（沿用既有过滤） | **无值域规则** | DOC 是目标变量，不因模型难以预测而删除 |

### 结果（\`scripts/audit_covariates.py\`）

771,581 行协变量候选中拒绝 3,522 行：

| 变量 | 行数 | 接受 | 拒绝 | 原因 |
|---|---:|---:|---:|---|
| temperature | 211,230 | 207,710 | 3,520 | above_physical_max 2,310；unsupported_unit(deg F) 1,210 |
| ph | 216,333 | 216,331 | 2 | above_physical_max（75、825） |
| spec_conductance | 254,044 | 254,044 | 0 | — |

处理记录：\`data/processed/covariate_quality_report.json\`（含规则、哈希、逐原因计数与原始样例）；
被拒绝的原始行：\`data/processed/covariate_rejections.csv\`。

### 新数据版本

\`data/processed/mississippi_graph_v05.pt\`（v04 未被覆盖），
同目录 \`mississippi_graph_v05.provenance.json\` 记录规则版本与摘要。

v04 → v05 的实际差异：

| 项 | 结果 |
|---|---|
| 水温通道 | **38 个单元改变**，最大值 1310 °C → 36.7 °C；规则外单元数 0 |
| DOC 标签 \`y\`、\`y_mask\` | **完全未变** |
| 流量、regime、edge_attr | 完全未变 |
| 站点数 / 月份数 / 边数 | 571 / 652 / 562，未变 |

## R2：补全依赖并验证干净检出

review 发现已提交的 \`GCNDocModel\` 与 \`TransportGCNImputer\` 都不接受 \`gate_mode\`，
干净检出无法构造 H3 模型。

处理方式**不是**打包提交未完成的动态门控工作，而是**移除这项依赖**：
H3 只用已发布的静态传输通路。已核对两种实现在 H3 所用配置下逐位相同：

| 核对项 | 结果 |
|---|---|
| \`GCNDocModel(architecture="transport_enc", env_encoder=True, env_groups=None)._build_inputs\` 的 6 个返回量 | 与 Git HEAD 版本**逐元素相同** |
| \`TransportGCNImputer(...)\` 在 \`manual_seed(7)\` 下的 state_dict | 与 HEAD 版本**逐元素相同**（23,461 参数） |
| 非 static 的 \`gate_mode\` | 直接 \`ValueError\` 拒绝，不会静默忽略 |

因此 H3 在提交版的 \`gcn.py/hydro.py\` 上可以直接构造、训练与恢复 checkpoint。

## R3：修正未运行的 E2a 与后续阶段控制

### R3a E2a

修订 1 的 \`e2a_partial\` 继承的是原 E2b 的外层 test 与 637 个未来 DOC context，
最终推理仍能看到未来 DOC，因此不是"无同期 DOC"的时间外推对照。

- 新增 \`e2a_strict\`：来源为冻结的 \`e2a_strict\` mask，沿用原 **3184** 个 test 目标，
  **无 \`context\` 角色**，val_context 记录但不在 \`visible_roles\` 中。
- 原 \`e2a_partial\` 更名为 \`e2b_partial_hiddenctx\`，明确标为"选择可见性诊断"，
  并从 8 个关键场景中移出。

### R3b 可见性显式化

每个 mask 现在存 \`visible_roles\`；训练器只按它开前向，
存在但未声明的角色保持关闭但**仍在 split 中**，观测单元不会无解释地消失。
声明 \`val\` 或 \`test\` 会被直接拒绝。

### R3c 扩展核验与阶段门禁

| 问题 | 修复 |
|---|---|
| \`verify --stage expand\` 按记录写入阶段过滤，会漏掉复用的 27 条 | 改为按阶段应有的 \`(arm, seed, mask)\` 网格选择；缺项、多项、重复都失败；结果里保存 config_hash 与 checkpoint 哈希 |
| \`audit_run\` 的 \`expected_config\` 未使用，调用方把记录自身哈希当预期值 | 新增 \`expected_config_for()\`：用**当前源码、协议、数据内容哈希与 mask 内容哈希**重算身份并逐字段比对；verify 与 frozen_eval 一律严格模式 |
| 外层 test 门禁只检查 pilot promote | 改为要求 T17 判定 promote + 标为 complete + 120 配置齐全 + 磁盘运行集合与判定记录的清单哈希一致；河网主张还需 T18，必须 \`--claim network\` |
| \`run_h3.py --stage expand\` 无门禁 | 代码级门禁：expand 需要 pilot 判定，T18 需要 expand 判定 |

严格身份核验在实施过程中**当场发现两个真实缺陷**，已修复：

1. \`identity()\` 把 mask 路径写成绝对路径，使同一配置在不同检出目录得到不同哈希；
   现统一存仓库相对路径。
2. 冒烟阶段的身份沿用完整 200 轮协议，而实际只训练 3 轮，
   记录里的 \`training\` 与实际不符；现按实际生效协议生成身份。

### R3d 截断说明

\`scripts/analyze_h3_pilot.py\` 不再写死"只有 ENV 被截断"这类结论。
截断单元数、比例、原始与截断后 log_RMSE、以及截断对相对优势与种子胜出计数的影响，
全部由逐运行预测动态生成，写入 \`clipping_report.json\` 与判定页。

## R4：新身份下重新确认 pilot

### 已完成

- 冒烟：\`experiments/h3a_smoke_v1r2/\`，3 条链路、各 3 epoch，全部通过严格身份审计；
  旧静态前向兼容测试 93 passed。见 \`smoke_report.json\`。
- pilot：27 次（3 臂 × 3 开发 mask × seed 0/1/2），\`experiments/h3a_v1r2/\`。
  修订 1 的 \`experiments/h3a_v1/\` 与 \`experiments/h3a_v1/pilot_decision.json\`
  完整保留，未改写。

### 结果

见 \`experiments/h3a_v1r2/pilot_decision.json\` 与 \`pilot_decision.md\`。
判定仍为 **promote**（达到 pilot 扩展标准），但收益幅度小于修订 1：

| 口径 | 修订 1（v04） | 修订 2（v05） |
|---|---:|---:|
| E2b H3A 相对 ENV | −7.21% | −4.64% |
| E2b H3A 相对 H2X | −5.69% | −3.50% |
| E2b 同时优于两对照的种子 | 2/3 | 3/3 |
| E2b ENV 平均原始 log_RMSE | 0.3696 | 0.2864 |
| E2b ENV 平均截断后 log_RMSE | 0.2999 | 0.2864 |

两个结论值得单独记录：

1. 修正异常水温后，ENV 的原始与截断后误差几乎重合（0.2864 / 0.2864），
   说明修订 1 里 ENV "原始 0.3696、截断后 0.2999" 的巨大落差主要由那一个
   异常单元驱动，而不是长尾生态变异。
2. H3A 的优势缩小但仍然存在，且**跨种子一致性提高**（2/3 → 3/3）。
   这与 review 中固定权重诊断的方向一致（该诊断预测 H3A 相对 H2X 约 −5.7%，
   本次重训实测 −3.50%；诊断不能替代重训，量级差异即来自重新选择权重）。

## 仍未解决

- WQP 缓存中有 57 条 DOC 标签为 0 mg/L；本轮未改动 DOC 标签（review 明确要求
  不因模型难预测而删除目标值），留待下一轮单独决定规则。
- 外层 test 仍未读取；\`scripts/run_h3_frozen_eval.py\` 已上锁，
  需要 T17 完成且清单一致。
- 运行身份记录工作树源码的 sha256，本机 \`core.autocrlf\` 为 true，
  跨机复算需先固定行尾。
- T17–T19 未执行。
