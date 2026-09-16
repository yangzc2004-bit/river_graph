# H3-A 修复复审响应（V1–V9）

日期：2026-09-16。来源：`docs/h3a_v1r2_review_20260916.md`。
本轮不改变任何冻结阈值，不读取外层 test。

## V1 资产独立备份（先做，因为它已经出过一次事）

新增 `scripts/backup_assets.py`：把 `data/`、`experiments/`、`src/`、`scripts/`、
`tests/`、`configs/`、`docs/`、`cache/` 与根目录小文件复制到独立目录，
逐文件记录 path/size/sha256，并对清单本身取摘要。

```
python scripts/backup_assets.py --target D:/river_graph_backup_20260916
python scripts/backup_assets.py --target ... --verify --restore-check 5
```

执行结果：

| 项 | 结果 |
|---|---|
| 文件数 / 体积 | 4,489 / 1,960.4 MB |
| 清单摘要 | `897d64730fae6d19` |
| 复核 | missing=0, corrupt=0 |
| 恢复抽查 | 5 个文件复制回临时目录后哈希一致，抽查通过 |

## V2 v05 恢复途径核查

在仓库、备份目录、系统临时目录中搜索 `mississippi_graph_v0*.pt`；
检查回收站（无大文件）与卷影副本（无可用快照）。**未找到 v05，也未找到 v04。**

因此按 review 的要求标注：

> `experiments/h3a_v1r2/` 的 27 条记录：**保存产物可复核，输入快照不可用。**
> 记录本身完整、自洽、已提交；无法在本机从 checkpoint 重新生成预测。

新增 `scripts/check_source_identity.py` 让这件事可判定而不是靠记忆：

```
python scripts/check_source_identity.py --root experiments/h3a_v1r2
{"reproducible": 3, "uncommitted": 2, "stale": 5, "missing": 0}
```

- `uncommitted`：`models/gcn.py`、`models/hydro.py` —— 工作区带未提交改动，
  与运行一致但干净检出不复现。
- `stale`：本轮修改过的 5 个文件 —— 运行无法再从当前源码重新推导。

## V3 历史覆盖损失的解释

### 根因

`fetch_station_results` 只检查响应是否以 CSV 表头 `Org_Identifier` 开头，
而**被截断的响应同样以表头开头**，于是残缺响应被当作完整历史缓存下来，
并在之后的每次数据构建中继续使用。

### 量化

新增 `scripts/audit_cache_completeness.py`，用 NWIS 系列目录（独立来源）
逐站核对缓存实际含有的 DOC 样本数：

| 项 | 值 |
|---|---:|
| 站点 | 571（缓存文件齐全） |
| 低于目录计数 | **48** |
| 缺失 DOC 样本合计 | **1,895** |

最严重的四站：06934500（2 / 432）、07144100（15 / 445）、
03303280（11 / 382）、07049690（0 / 342）。前四站合计 1,573，占 83%。
这正好解释旧观测集合与 v06 之间 955 个 station/month 的缺口。

### 对照下载与恢复

对 48 个可疑站点重新下载到**新目录** `data/raw/wqp_control`，
逐条记录请求 URL、时间、HTTP 状态、字节数与 sha256
（`control_download_log.json`）。结果：

- 7 个站点的对照副本更完整，按实测内容提升（`cache_promotion_log.json`）。
- 3 个站点的对照副本**更少**（如 05455100 38→3、445548111032200 19→0），
  因此提升是逐站按内容决定的，不是"新的更好"。
- 缺失样本合计 **1,895 → 525**。

### 残余

- 站点 **07144100** 即使反复全新下载仍只返回约 15 条，而目录记录 445 条。
  这不是本地缓存问题，需要向提供方查证；已作为未解决项保留。
- 其余 40 站缺口多为 1–16 条，与提取器的 fraction/unit/detection 过滤一致
  （例如 669 / 683），属于正常差异而非截断。

### 代码修复

| 位置 | 修复 |
|---|---|
| `wqp._payload_looks_complete` | 校验 Content-Length 已完整送达，且 CSV 记录结构闭合（末行字段数与表头一致） |
| `fetch_station_results(min_doc_samples=...)` | 语义检查：响应或既有缓存若明显少于独立期望值则重取；仍不足时保留最长响应并明确报告 |
| `build_dataset.acceptable_doc_samples` | 阈值放在"正常过滤差异"与"截断"之间，避免对提供方本就少返的站点反复重试 |
| `build_dataset.raw_input_manifest` | 按版本记录每个缓存文件的大小、sha256、实际 DOC 数与目录计数，构建时打印告警 |

### 结果：v07

用恢复后的缓存重建为 **v07**（v06 已登记，不再复用该名字）：

| 项 | v06 | v07 |
|---|---:|---:|
| 观测单元 | 32,420 | 33,465 |
| 有 DOC 观测的站点 | 567 | **571** |
| 标签覆盖率 | 8.7% | 9.0% |
| mask 迁移丢失单元 | 955 | **0** |

v07 下重新迁移冻结 mask，**所有角色零丢失**：
e1 train/val/test 恢复为 23794/2644/6610，e2b 为
25771/3274(+819)/2547(+637)，与冻结 mask 的站点-月份分配逐一致。

## V4 数据版本发布保护

`scripts/build_dataset.py`：

- `--version` 改为**必填**，默认值 `v05` 已移除。
- 新增 `data/processed/dataset_registry.json` 登记每个已发布版本的
  sha256、字节数、发布时间、git 修订与全部溯源字段。
- **已登记版本永不改写**：即使文件已被删除，重新用同名发布也会拒绝
  （这正是 v05 的情形）。
- 未登记但已存在的同名文件：默认拒绝，需显式 `--allow-republish`。
- 发布走"临时文件 → 校验哈希 → 原子替换 → 登记"，不会留下半套版本。
- 质量报告、拒绝记录、原始输入清单全部**按版本命名并登记哈希**：
  `covariate_quality_report_<v>.json`、`covariate_rejections_<v>.csv`、
  `raw_input_manifest_<v>.json`。

## V5 流量 QC 与真实构建路径一致

- `quality.apply_rules` 现在对 `signed` 变量检查**绝对值**：
  `-4e6` 与 `+4e6` 都被拒绝，`-3150`（潮汐反向流）仍被接受。
- 抽出 `build_dataset.prepare_discharge()`，在真实构建路径上调用同一套规则，
  再按月聚合。v07 构建中实际执行：候选 5,978,400，接受 5,978,400，拒绝 0。
- 报告中的候选数/接受数/拒绝数与逐行理由来自这次真实执行，不再是空口声明。

## V6 旧 mask 的源网格身份检查与迁移

- 新增 `experiments/masks/source_grid.json`：声明冻结 mask 的网格
  （571 站 × 652 月 × 33048 观测）与**站点顺序摘要**。
- `build_masks` 先校验源网格与站点顺序，不一致直接拒绝；
  月份数不同则按 station/month 解码再编码迁移，
  对每个角色报告 `before/after/dropped_unobserved`。
- 外层 test 的比对改为在 (station, month) 坐标下进行；
  迁移**不允许凭空产生**任何 test 单元。
- review 复现的 `cell 137646 in both val and test` 在 v06 上同样出现，
  迁移后消失；v07 上零丢失。

## V7 T19 network 门禁与导出

原实现把 no-message 记录数与 120 条三臂网格比较，且导出选集只含三臂。

- 新增独立的 40 条网格：`task_grid("no_message")` = 8 关键场景 × 5 种子 × 1 arm。
- `scripts/run_h3.py --stage no_message` 有独立阶段与门禁（需 T17 判定）。
- 分析脚本新增 `no_message` 阶段，产出 `no_message_decision.json`
  （含它自己的 40 条清单哈希）与四臂对照页。
- `unlock()` 现在：要求 T17 判定；核对 120 条集合与清单哈希；
  申请 network 时另行要求 T18 判定、核对 **40/40** 完整性与清单哈希，
  并逐条做严格身份核验。
- 导出选集 = 三臂 120 + 对照 40，对照确实进入导出。
- 新增门禁测试：完整 120+40 通过、缺 1 条对照被拒（提示 of 40）、
  对照清单变化被拒、无 T18 判定被拒、对照进入导出选集。

## V8 可独立运行源码的判定

`scripts/check_source_identity.py` 把"这个运行能不能在干净检出里复现"
变成一个可执行的检查（reproducible / uncommitted / stale / missing），
并在非全绿时返回非零，可用作发布门禁。本轮生成运行的三个哈希
（recorded / working / committed）都记录在 `source_identity.json`。

## V9 新身份下的 smoke 与 pilot

```
python scripts/run_h3.py --stage smoke --workers 3      # experiments/h3a_smoke_v1r3
python scripts/run_h3.py --stage pilot --workers 7      # experiments/h3a_v1r3
```

冒烟：3 条链路，各 3 epoch，严格身份审计通过，无 NaN/泄漏，
旧静态前向兼容测试 93 passed（`smoke_report.json`）。

pilot：27 次（3 臂 × 3 开发 mask × seed 0/1/2），全部通过严格身份核验。
判定仍为 **promote**（`experiments/h3a_v1r3/pilot_decision.json`）：

| 口径 | 修订 1（v04） | 修订 2（v05） | 修订 3（v07） |
|---|---:|---:|---:|
| E2b H3A 相对 ENV | −7.21% | −4.64% | **−5.51%** |
| E2b H3A 相对 H2X | −5.69% | −3.50% | **−2.03%** |
| E1 MAE 相对 H2X | −0.23% | −0.69% | −0.21% |
| E3 MAE 相对 H2X | −1.92% | +0.29% | −2.47% |
| E2b 同时优于两对照的种子 | 2/3 | 3/3 | **3/3** |
| 截断单元（env/h2x/h3a） | — | 3/10/7 | 6/12/15 |

必须如实说明两点：

1. **相对 H2X 只有 −2.03%，紧贴 −2% 阈值，没有余量。**
   判定按冻结规则成立，但不宜表述为稳健优势。判定页已自动附上该提示。
2. **E3 的内部验证站集合本修订发生了变化**（恢复缓存后合格站点集合改变，
   `e3_internal_*` 的 val 数从 2694/2563/2480 变为 2305/3416/2051），
   因此 E3 的跨修订比较不是同集合比较，E3 的 MAE 水平也从约 1.6 升到约 2.0。

## 仍未解决

1. **v05 与 v04 输入快照不可恢复**；r2 的 27 条只能作为保存结果复核。
2. **07144100** 的目录/实际覆盖差异需要提供方查证。
3. **`gcn.py`/`hydro.py` 仍未提交**：本机生成的运行在干净检出上会有两个
   源码哈希不一致。已可被 `check_source_identity.py` 明确报出。
4. 外层 test 仍未读取。
5. T17/T18/T19 未执行。
