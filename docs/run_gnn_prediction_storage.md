# 预测保存与缓存使用说明（A1）

对应 `docs/execution_plan_20260912.md` 的 A1 阶段。本文件说明**怎么用**新的保存/缓存行为，以及**哪些事情它还不能保证**。

## 1. 为什么需要这份说明

`run_gnn.py` 的缓存判定原来是「指标 JSON 里有这个 mask 就跳过」。这会产生一个假完成：传入 `--save-predictions` 但 parquet 不存在时，程序只打印 `cached, skip`，既不报缺失也不补保存。A0 清点发现，正是这个行为留下了 **205** 个「有指标、无逐样本预测」的组合（其中 120 个是新五种子的 key mask）。

现在缓存判定改为检查**磁盘上实际存在什么**，并对缺失给出明确状态。

## 2. 缓存状态与行为

每个 (模型, mask) 组合会落入四种状态之一：

| 状态 | 含义 | 默认行为 |
|---|---|---|
| `complete` | 指标 + 预测都在 | `cached, skip`，**不训练** |
| `metrics_only` | 有指标、**没有**逐样本预测 | 见下表，**默认不重训** |
| `predictions_only` | 有预测、没有指标条目 | 从预测的 test 行复算指标补齐 JSON，**不训练** |
| `absent` | 两者都没有 | 正常训练一次，并存指标（带开关时也存预测） |

`metrics_only` 的三种处理：

| 你传入的开关 | 行为 |
|---|---|
| `--rebuild-predictions` | **显式重跑该 mask 一次**，指标与预测来自同一次运行，并写 provenance sidecar（该开关**隐含** `--save-predictions`） |
| `--save-predictions` | **报告缺失**（`PREDICTIONS MISSING ...`），计入末尾汇总，结束退出码 **2**；**不训练** |
| 都不传 | 复用已有指标，打印 `cached metrics, skip` |

要点：

- **缺失永远不会被宣告为成功。** 请求保存而预测不存在时，进程以非零码结束并逐条列出缺失组合。
- **不会自动重训。** 只有显式给 `--rebuild-predictions`（或 `--force`）才会训练。这一点是刻意的：分析阶段跑一次入口脚本不应该悄悄消耗算力。
- **训练必然落盘。** `--rebuild-predictions` 与 `--force` 会**强制打开**保存开关：两个开关都会触发训练，而训练不保存预测正是本流程存在的意义所在，所以不会被静默允许。
- **身份冲突返回非零码（3）。** 全部运行都因身份不符被拒绝时，进程不会以 0 退出，自动化不会误判成功。
- 现有开关只保证**携带它的新训练**走保存路径，**不代表旧实验已经归档**。旧结果是否完整，用下面的审计命令查。

## 3. 审计：先看缺口，再决定要不要训练

```bash
python scripts/run_gnn.py --verify-predictions
python scripts/run_gnn.py --verify-predictions --only H2Xf   # 只看某模型
```

该模式**完全不训练**，只读 `experiments/results/*.json` 与 `experiments/predictions/`，输出「有指标、无预测」的组合清单，存在缺口时退出码 1。当前仓库实测：**205** 个缺失组合、78 个完整组合。

逐文件的哈希与指标一致性核验用 A0 的脚本：

```bash
python scripts/audit_artifacts.py --verify      # 身份 + 复算 vs 冻结表
python scripts/build_prediction_manifest.py     # 重建本轮分析输入白名单
```

## 4. 未来正式训练：`run_ladder.py`

`run_gnn.py` 的 `--save-predictions` **仍是显式开关**（下文 §6 说明原因）。新的正式训练走包装入口，它**强制打开保存**：

```bash
python scripts/run_ladder.py --arch directed --seed 0 \
    --model-name H1_directed_river_s0 --only e2b_partial
python scripts/run_ladder.py --arch transport_enc \
    --dataset data/processed/mississippi_graph_v04.pt \
    --model-name H2X_v2 --dry-run
```

- `--dry-run` 打印将要使用的模型名、`config_hash` 和每个 mask 的缓存状态（含冲突提示），**不加载数据、不训练**。
- 其他参数原样转发给 `run_gnn.py`。
- 如果某次运行会覆盖**配置哈希不同**的已有预测，默认拒绝（退出码 3），需要 `--force` 明确确认。

## 5. 身份与中断保护

**原子写入。** parquet 与 sidecar 都先写入同目录的 `.tmp`，`flush + fsync` 后用 `os.replace` 就位。因此：

- 中断的写入留下的是 `.tmp`，不会被识别为完整缓存（`cache_state` 只看最终文件名）。
- 不存在「写了一半的 parquet 被当成有效结果」的情况。

**但单文件原子不等于整次运行一致。** parquet、sidecar、指标 JSON 是三个独立替换点，没有共同的完成标识，所以一次运行可能在写完预测、更新指标之前中断，留下「新预测 + 旧指标」同时存在的状态。

因此**缓存命中时会校验一致性**：从已存预测复算**全部指标字段**（`mae/r2/rmse/log_mae/log_r2/log_rmse/pbias/n`，不只是前三个），与指标 JSON **以及 sidecar 里记录的 `metrics`** 比较。差异被判定为**不一致**，并且**默认拒绝继续**：

- 不能"从预测修复指标"就了事。中断的那次运行用的是**什么配置没人记录**，把它的输出按当前配置的身份报出去，等于把别人的结果记在自己名下。这正是本项目存在的意义所在。
- 处理方式：`--rebuild-predictions`（重训并同时重写两者，退出码不变）；`--force`（同上，并可覆盖身份不符）；或 `--repair-metrics`（明确表示**信任预测**，用它重写指标）。三者都要显式给出。
- 不一致时进程以退出码 **4** 结束，并逐字段打印差异。
- 只有**遗留条目**（无 sidecar，生成时还没有这套机制）才自动修复，且会明确打印为 `METRICS REPAIRED ... (no provenance sidecar...)`。

**身份校验。** 每次保存同时写 `{model}__{mask}.meta.json`，内容包括：

- `config_hash`：由**训练种子、架构、variant、lr、weight decay、edge dropout、share_weights、env_groups、env_encoder、模型名**，以及**数据集与 mask 的内容 sha256** 共同决定；
- `dataset` / `mask`：路径 + 字节数 + mtime + sha256；
- `split_sizes`、`caller`、`created_at`、`python` 版本、`results_path`；
- `metrics`：本次运行在该 mask 上的完整指标记录，供以后核对「预测与指标是否同源」；
- `export_scope`：说明导出范围。

**为什么要包含内容哈希**：只看路径和文件名不足以判断身份。此前 `config_hash` 不含数据集/mask 身份，因此**就地修改数据集或重新生成 mask 后，程序仍然返回 `cached, skip`**，等于用旧数据划分的结果冒充新配置。现在改动数据集或 mask 的内容都会改变身份，复用会被拒绝并逐字段说明差异（例如 `dataset_sha256: stored=... current=...`）。

重新保存时如果 sidecar 记录的配置与当前不一致，`save_predictions` 抛 `PredictionConflictError`，运行器报 `IDENTITY MISMATCH` 并跳过，进程以退出码 **3** 结束。这样「同名不同配置」不会静默覆盖历史结果；要给新配置独立名字，用 `--model-name`。

**退出码汇总**

| 码 | 含义 |
|---:|---|
| 0 | 正常完成（含"全部命中缓存"） |
| 1 | `--verify-predictions` 发现「有指标、无预测」 |
| 2 | 请求保存但预测缺失（未训练） |
| 3 | 身份不符，拒绝复用 |
| 4 | 缓存条目的指标与预测不一致，拒绝猜测 |

**文件名约定保持不变。** 仍是 `{model}__{mask}.parquet`，因为 `recompute_metrics.py`、`run_freeze.py` 等既有工具按该约定解析。身份信息由 sidecar 承担，不改名、不重命名现有 85 个文件。

**关于历史文件。** `experiments/predictions/` 中现有 85 个预测**没有** sidecar（它们生成时还没有这套机制）。运行器把它们记为 `legacy file`，属于「身份已知程度有限」，不是错误；不要据此宣称可完整复现。

## 6. 已知限制（不要过度声明）

- `--save-predictions` 仍需主动启用；旧实验不会因为代码修好就自动获得预测。
- 导出的是**有真实 DOC 标签的观测单元**（`y_mask` 为真的单元），**不是** 571×652 完整网格的缺测填补产品。完整缺测网格导出是另一条流程，尚未实现。
- `train` / `context` 单元可能看到自身 DOC 值；诊断与残差分析**只能使用合法的 test 单元**（parquet 的 `split` 列，A0 已抽检确认与 mask 的 test 集完全一致）。
- 仍**没有**权重保存、优化器状态、完整环境锁定；有配置/数据/mask 哈希，但没有「一键重放得到同样数字」的完整闭环。不要把这个流程称为已闭环的可复现实验管理。
- H1/H2/H2X 五个训练种子的逐样本预测**依然不存在**。`--rebuild-predictions` 可以用来补，但那会训练 120 次；这属于 B 阶段，需要先定预算。
- `recompute_metrics.py` 现在会先报告「重写会丢多少行 / 改多少行」并默认拒绝写入：当前 parquet 集只覆盖 154 行冻结表中的 85 行（69 行无预测、1 行来自另一次运行），直接重算会造成静默数据丢失。
- **分析侧的预测来源选择**由 `src/river_graph/experiments/prediction_sources.py` 负责：清单按 `(batch 目录, 文件名, sha256)` 登记，**每个 (模型, mask) 只选一个来源**。这是必需的——被排除的冲突文件与恢复副本**同名**，只按文件名放行会让两者都被读取、把测试单元重复计数。

## 7. 清单的内容哈希棘轮

`experiments/frozen_results/prediction_manifest_*.json` 是分析输入的白名单。它有一个**必须**保留的性质：**不能靠重新哈希磁盘内容来重建**。否则把某个预测覆盖掉、再跑一次生成脚本，新内容就会被当成可信输入重新放行——白名单就形同虚设。

因此每个获准条目都带一个 `pinned_sha256`，从上一版清单**继承**：

- 字节与 pin 一致的条目正常放行；
- 字节已变的条目**永不重新放行**，记为 `content_changed_since_pin`，并把「pin 值」与「磁盘当前值」一起写进清单，旧 pin **保留**，方便对照；
- 要接受新字节必须显式 `--repin`（会打印每一处变化）。人工编辑清单里的 `sha256` **无法**绕过：`load_sources` 以 `pinned_sha256` 为准，不以 `sha256` 为准。
- 停止存在的文件，其 pin 记为 `pins_removed`。

`load_sources()` 同时报告 `missing`、`hash_changed`、`pin_mismatch`、`duplicate_keys`、`unmanifested_on_disk`（磁盘上有但清单里没有的文件，**永不使用**）。
