# Phase 0：现有项目审计报告

> 审计范围：`ECC/`（第一轮实验代码）+ 直接上游依赖（`task1_weight_importance/`、
> `llm_weight_fault_lab/mauep_v0_4|v0_5|v0_6`、`BCH/`、`Hamming/`）
> 审计时间：2026-11-14
> 原则：**只读不改**。本文件不修改任何源码。

---

## 1. 代码清单

### 1.1 本轮 ECC 实验代码（`ECC/`）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `allocation/build_level_map.py` | 118 | 全局 Smax 排序 → High/Medium/Low 等级图 |
| `ecc/bch_high.py` | 45 | 包装现有 `BCHCode(5, x⁵+x²+1, t=3)` = BCH(31,16,7) |
| `ecc/bch_medium.py` | 116 | BCH(24,16) 删余码：校验矩阵、综合征表、译码 |
| `ecc/low_none.py` | 24 | Low-0 透传 |
| `ecc/level_codec.py` | 33 | 三级调度接口 |
| `ecc/anomaly_guard.py` | 58 | 逐 Tensor 幅值检测与置零 |
| `simulation/packed_codec.py` | 120 | 码字打包 uint32 + 综合征查表（向量化译码） |
| `simulation/apply_ecc.py` | 221 | 编码→注错→译码→恢复（BERT 模式） |
| `simulation/fixed_errors.py` | 160 | 固定错误数模式 |
| `evaluation/evaluate_ppl.py` | 107 | NLL/PPL 评估 |
| `evaluation/tensor_thresholds.py` | 68 | 离线阈值统计 |
| `evaluation/wikitext_loader.py` | 15 | 直接读 HF arrow 缓存 |
| `evaluation/zero_weights.py` | 75 | 误杀代价实验 |
| `evaluation/run_experiments.py` | 38 | 批量跑 BER 网格 |
| `tests/run_functional_tests.py` | 115 | ECC 功能穷举测试 |
| `tests/test_anomaly_guard.py` | 45 | 异常检测单元测试 |

**合计 16 个文件、约 1,320 行。**

### 1.2 上游依赖

| 目录 | 规模 | 与本系统的关系 |
|---|---|---|
| `task1_weight_importance/` | 750 行 / **4.46 GB 数据** | 提供 `I_max_log10.npy`（= log10 Smax），**等级分配的唯一输入** |
| `llm_weight_fault_lab/mauep_v0_5/` | 745 行 / **16.34 GB 数据** | Bit-level 风险地图（`risk_log10.npy`），Sm​​ax 的原始来源 |
| `llm_weight_fault_lab/mauep_v0_4/` | 1,315 行 | 提供 `gram/*.npz`（F、C），本系统**未直接使用**，但 V0.5 依赖 |
| `llm_weight_fault_lab/mauep_v0_6/` | 1,265 行 | 理想保护分析（独立研究线，非本系统组成部分） |
| `BCH/` | 7,074 行 | BCH 母码、缩短码、求解器；High 直接复用 |
| `Hamming/` | 2,859 行 | 第一轮实验未使用（V0.7 曾用） |

### 1.3 实验产物

| 路径 | 内容 | 大小 |
|---|---|---:|
| `ECC/outputs/allocation/` | 35 个 `level_map.npy` + 汇总 | 223 MB |
| `ECC/outputs/thresholds/` | 35 个阈值 JSON/CSV | < 1 MB |
| `ECC/outputs/model_results/` | 48 个结果 JSON | 1.3 MB |
| `ECC/outputs/ecc_tests/` | 功能测试明细与汇总 | < 5 MB |

---

## 2. 功能清单与实现状态

| # | 功能 | 实现位置 | 状态 |
|---|---|---|---|
| 1 | Bit-level 风险分析 | `mauep_v0_5` | ✅ 已完成（外部依赖） |
| 2 | Smax 全局排序 | `build_level_map.py` | ✅ |
| 3 | High/Medium/Low 分配 | 同上 | ✅ |
| 4 | BCH(31,16,7) 编解码 | `bch_high.py` + `packed_codec.py` | ✅ |
| 5 | BCH(24,16) 删余编解码 | `bch_medium.py` + `packed_codec.py` | ✅ |
| 6 | 随机 bit error 注入 | `apply_ecc.py` | ✅ |
| 7 | ECC 译码与权重恢复 | `apply_ecc.py` | ✅ |
| 8 | Anomaly Containment | `anomaly_guard.py` | ✅ |
| 9 | NLL/PPL 评估 | `evaluate_ppl.py` | ✅ |
| 10 | 多 BER / 多 seed / 固定错误数 | `run_experiments.py`、`fixed_errors.py` | ✅（部分） |
| 11 | 结果统计与报告 | 手工脚本 + Markdown | ⚠️ **半自动** |

---

## 3. 调用关系（现状）

```
task1_weight_importance/outputs/weight_importance/*/I_max_log10.npy
        │
        ▼
allocation/build_level_map.py ──► ECC/outputs/allocation/*/level_map.npy
        │
        ▼
evaluation/evaluate_ppl.py
        │  import
        ▼
simulation/apply_ecc.py ──► simulation/packed_codec.py ──► ecc/bch_high.py
        │            │                                     └─► ecc/bch_medium.py
        │            └─► ecc/anomaly_guard.py ──► outputs/thresholds/tensor_thresholds.json
        │
        ▼
evaluation/wikitext_loader.py ──► HF datasets 缓存（arrow）
```

`simulation/fixed_errors.py` **另起一套张量遍历与等级分派逻辑**，与 `apply_ecc.py` 平行。

---

## 4. 输入输出格式

| 数据 | 格式 | 生产者 | 消费者 |
|---|---|---|---|
| `I_max_log10.npy` | float32 `[num_weights]` | task1 | 等级分配、分数查询 |
| `level_map.npy` | uint8 `[num_weights]`（0/1/2） | build_level_map | apply_ecc、fixed_errors、zero_weights |
| `tensor_thresholds.json` | `{"thresholds": {"L0.q_proj": 1.23, ...}}` | tensor_thresholds | anomaly_guard |
| 结果 JSON | 自定义 schema（无版本号） | 各脚本 | 人工 / 汇总脚本 |

---

## 5. 发现的问题

### 5.1 架构问题

| 编号 | 问题 | 影响 |
|---|---|---|
| **A1** | **层次倒置**：`evaluation/` 导入 `simulation/` 的内部函数（`apply_to_model`、`BLOCKS`、`get_module`） | 评估与仿真无法独立替换 |
| **A2** | **重复实现**：`apply_ecc.py` 与 `fixed_errors.py` 各有一套张量遍历 + 等级分派 + 计数（约 100 行重复） | 改一处漏一处（已发生：guard 字段只加进了一半） |
| **A3** | **无统一入口**：6 个脚本各自 argparse，参数名不统一（`--strategy` vs `--target`） | 用户必须知道跑哪个脚本 |
| **A4** | **无配置层**：模型路径、目标 Block、比例、BER 全靠命令行默认值 | 无法固化为"版本配置" |
| **A5** | **无实验 ID / 元数据**：结果只按文件名区分，没有配置快照、时间、代码版本 | 不可追溯 |
| **A6** | **路径耦合仓库布局**：`ROOT/"models"`、`ROOT/"BCH"`、`ROOT/"llm_weight_fault_lab"` 等 12 处 `sys.path.insert` | 无法在新目录独立部署 |

### 5.2 正确性/一致性问题

| 编号 | 问题 | 证据 |
|---|---|---|
| **B1** | **两个基线常数并存**：`evaluate_ppl.py` 用 `REFERENCE_NLL = 2.439669`，`fixed_errors.py` / `zero_weights.py` 用 `2.439884` | 实测 grep 命中 4 处 |
| **B2** | `fixed_errors.py` 的结果 JSON **缺 `anomaly_guard` 字段** | 已实测确认 |
| **B3** | `run_experiments.py` **不支持 `--anomaly-guard`** | 需手工逐条调用 |
| **B4** | 数据集加载**硬依赖 HF 缓存目录结构**（`Path.home()/.cache/huggingface/...`），且写死哈希目录名 | 换机器即失效 |

> B1 的后果：同一次实验在不同脚本里算出的 ΔNLL 相差 0.000215，
> 曾导致我在汇总表中出现与运行日志不一致的数字。

### 5.3 数据与可复现性问题

| 编号 | 问题 |
|---|---|
| **C1** | 16.34 GB 风险地图 + 4.46 GB 重要性数组是**外部前置数据**，没有校验和、没有 manifest、没有版本号 |
| **C2** | 结果 JSON 无 `schema_version`，字段随脚本改动漂移（B2 即例证） |
| **C3** | 没有记录 Git commit / 环境版本 |
| **C4** | `apply_to_model` **就地修改模型权重**，依赖"模型对象一次性使用"；无显式备份/恢复 API |

### 5.4 未实现功能（相对任务书要求）

- 统一 CLI（`run.py`）
- 统一配置（YAML）
- 实验注册表（experiment ID → 配置 + 结果）
- 自动报告生成
- 独立安装（`pyproject.toml`）
- `pip install` 后可用
- 端到端测试套件（现有仅功能测试与 guard 单测）

---

## 6. 风险清单

| 风险 | 说明 | 建议 |
|---|---|---|
| **R1** | 重构时误改冻结算法（BCH 参数、删余位置、比例） | 重构前后**逐位对比**冻结产物（等级图、码字、统计） |
| **R2** | 数据迁移丢失（16 GB 风险地图） | 只**引用**不复制；用配置指向原路径，并记录校验和 |
| **R3** | 双基线常数导致历史结果不可比 | 统一为"运行时计算的 clean baseline"，历史 JSON 保留原文不动 |
| **R4** | 内存限制（本机可用提交内存约 2 GB） | 坚持流式/分块；禁止全量加载风险地图 |
| **R5** | 重跑代价高（单次模型评估约 5 分钟） | 端到端验收用小子集，正式回归用完整配置 |

---

## 7. 审计结论

1. **功能齐全但工程化程度低**：11 项能力全部具备，但没有统一入口、配置、元数据与报告。
2. **冻结算法本身是干净的**：BCH 参数、删余位置、比例、阈值规则都与任务书一致，
   本次审计**未发现算法层偏差**。
3. **主要问题是架构与一致性**（A1–A6、B1–B4），不是算法。
4. **可以安全重构**：现有代码全部保留不动，新系统作为独立目录建立，
   通过"冻结产物对比"证明等价性。

---

## 8. 建议的重构边界

**保留不动**：
- `BCH/`、`Hamming/`（母码实现，只读复用）
- 16.34 GB 风险地图、4.46 GB 重要性数组（只读引用）
- 所有历史结果 JSON / CSV（作为回归基准）

**新建**：
- `MA-UEP/` 独立项目（源码 + 配置 + 测试 + 文档）
- 通过配置指向上述只读数据

**废弃（不删除，仅停用）**：
- `ECC/` 下 6 个零散脚本，其逻辑迁入新系统并被测试覆盖
