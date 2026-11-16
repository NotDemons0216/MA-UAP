# Phase 1：架构设计

> 目标：把审计中识别的问题（A1–A6、B1–B4、C1–C4）逐条解掉，
> **不改动任何冻结算法**，并让历史结果可被逐位复现验证。

---

## 1. 设计原则

| 原则 | 具体做法 |
|---|---|
| **算法冻结** | BCH 参数、删余集合 Q、等级比例、阈值规则写成常量并加"冻结测试"锁死 |
| **只读引用大数据** | 16.34 GB 风险地图与 4.46 GB 重要性数组不复制，配置指向原路径 + SHA-256 |
| **单入口** | 所有操作经 `run.py`，内部模块不直接暴露给用户 |
| **配置驱动** | 一次实验 = 一份 YAML；代码里不出现模型路径、比例、BER |
| **可追溯** | 每次实验生成 `experiment_id`，保存配置快照、环境、数据指纹、结果 |
| **分层解耦** | 依赖方向单向：`evaluation → pipeline → {ecc, channel, protection, uep} → importance → model` |

---

## 2. 目录结构

```text
MA-UEP/
├── run.py                      # 唯一入口
├── pyproject.toml
├── requirements.txt
├── README.md
├── configs/
│   ├── v1_default.yaml         # 冻结配置（模型/比例/码型/阈值路径）
│   └── experiment_grid.yaml    # 批量实验网格
├── src/ma_uep/
│   ├── model/                  # Module 1
│   │   ├── loader.py           #   加载 / BF16 检查 / 结构枚举
│   │   └── evaluator.py        #   ★ 与 evaluation 的关系见 §4
│   ├── importance/             # Module 2
│   │   ├── reader.py           #   读 I_max_log10 / risk_log10（memmap）
│   │   └── ranker.py           #   全局 Smax 排序 + 分位阈值
│   ├── uep/                    # Module 3
│   │   ├── allocator.py        #   High/Medium/Low 分配
│   │   └── stats.py            #   码率 / 冗余 / 校验位统计
│   ├── ecc/                    # Module 4
│   │   ├── bch_high.py         #   包装 BCHCode(5, x⁵+x²+1, 3)
│   │   ├── bch_medium.py       #   删余码（Q={0,2,4,6,8,10,12,14}）
│   │   ├── low_none.py
│   │   ├── level_codec.py      #   调度
│   │   └── packed.py           #   向量化打包编解码
│   ├── channel/                # Module 5
│   │   ├── ber.py              #   BER 注入
│   │   └── fixed.py            #   固定错误数注入
│   ├── protection/             # Module 6
│   │   └── anomaly_guard.py
│   ├── pipeline/               # 编排层（新增，解决 A2）
│   │   ├── tensor_pass.py      #   ★ 唯一的张量遍历 + 等级分派实现
│   │   └── runner.py           #   单次实验的 12 步流程
│   ├── evaluation/             # Module 7
│   │   └── perplexity.py
│   └── experiment/             # Module 8
│       ├── config.py           #   配置加载与校验
│       ├── registry.py         #   experiment_id 生成与索引
│       └── reporter.py         #   Markdown/CSV 汇总
├── scripts/                    # 薄封装，仅调用 src（不写业务逻辑）
├── tests/
├── outputs/
│   ├── experiments/<EXP_ID>/
│   ├── reports/
│   └── logs/
└── docs/
```

---

## 3. 关键接口

### 3.1 数据契约（解决 C1/C2）

```python
@dataclass(frozen=True)
class TensorSpec:
    block_id: int
    module: str          # q_proj ... down_proj
    n_weights: int
    shape: tuple[int, int]

@dataclass(frozen=True)
class ImportanceSource:
    smax_path: Path          # I_max_log10.npy
    risk_map_dir: Path       # 可选，用于 b14 诊断
    sha256: str              # 记录指纹
    n_tensors: int
    coverage_note: str       # "35 tensors of 5 blocks, NOT full model"
```

**等级图**沿用现有格式 `level_map.npy`（uint8，0/1/2），保证与历史产物逐位可比。

### 3.2 唯一张量遍历（解决 A2）

```python
def run_tensor_pass(model, levels_by_tensor, codec_set, channel, guard, *, mode):
    """所有实验模式的唯一张量循环。mode ∈ {clean, no_protection, ma_uep, random_uep}"""
```

`apply_ecc` 与 `fixed_errors` 的重复逻辑合并到这里，BER 与固定错误数只通过 `channel` 策略区分。

### 3.3 ECC 接口（解决 R1）

```python
class Codec(Protocol):
    level: str
    k: int
    n: int
    parity_bits: int
    def encode(self, data_u16: np.ndarray) -> np.ndarray: ...
    def decode(self, words: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def data_of(self, words: np.ndarray, status: np.ndarray) -> np.ndarray: ...
```

三种实现（High / Medium / Low）满足同一协议；**冻结参数写成模块级常量**：

```python
HIGH_PARAMS = {"m": 5, "primitive_poly": [1,0,1,0,0,1], "t": 3}   # BCH(31,16,7)
MEDIUM_KEEP_PARITY = (0, 2, 4, 6, 8, 10, 12, 14)                  # 冻结
LEVEL_RATIOS = {"high": 0.05, "medium": 0.15, "low": 0.80}        # 冻结
CHANNEL_EXCLUDED_BF16_BITS = (14,)                                # 冻结（b14）
```

### 3.4 单次实验流程（解决 A3/A5）

```python
class ExperimentRunner:
    def run(self, cfg: ExperimentConfig) -> ExperimentResult:
        # 1 load model → 2 load importance → 3 rank → 4 allocate
        # 5 encode → 6 inject → 7 decode → 8 guard → 9 reload
        # 10 evaluate → 11 collect stats → 12 write + report
```

`ExperimentResult` 落盘为：

```text
outputs/experiments/<EXP_ID>/
├── meta.json          # 时间、模型、数据指纹、Git、环境、种子、BER、模式
├── config.snapshot.yaml
├── uep_stats.json
├── channel_stats.json
├── ecc_stats.json
├── guard_stats.json
├── eval.json
└── result.json        # schema_version=1.0
```

### 3.5 基线口径统一（解决 B1）

**取消所有硬编码 NLL 常数。** 基线由系统在实验开始时**实测**（clean 模型跑一次
WikiText-2），写入 `meta.json`，所有 ΔNLL 用同一个值计算。

历史报告中的 `2.439669` / `2.439884` 差异会在回归报告中作为已知差异说明，
**不修改历史数据**。

### 3.6 数据集加载（解决 B4）

三级回退：
1. 配置中的 `dataset_path`（本地 token 文件或 arrow 路径）；
2. `datasets` 库标准加载；
3. 直接读 HF 缓存的 arrow（现有 `wikitext_loader` 逻辑）。

并记录实际使用的来源与 token 数。

---

## 4. 模块依赖方向

```
experiment (config, registry, reporter)
      │
      ▼
pipeline (runner, tensor_pass)
      │
      ├──► ecc ──► (只读引用 BCH/ 母码)
      ├──► channel
      ├──► protection (anomaly_guard)
      ├──► uep ◄── importance
      └──► evaluation ◄── model
```

**禁止反向依赖**：`ecc/` 不得 import `evaluation/`；`model/` 不得 import `ecc/`（解决 A1）。

---

## 5. 配置方案

```yaml
# configs/v1_default.yaml
schema_version: "1.0"
model:
  path: models/Qwen2.5-1.5B-Base        # 相对仓库根，可被 CLI 覆盖
  dtype: bfloat16
  device: cuda:0
dataset:
  name: wikitext-2-raw-v1
  split: test
  stride: 128
  max_length: 256
scope:
  blocks: [0, 2, 15, 23, 27]
  modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  coverage_note: "35 tensors / 5 blocks, not full model"
importance:
  smax_source: task1_weight_importance/outputs/weight_importance
  ranking: global
uep:
  ratios: {high: 0.05, medium: 0.15, low: 0.80}
ecc:
  high: {type: bch_31_16_7}
  medium: {type: pruned_bch_24_16, keep_parity: [0,2,4,6,8,10,12,14]}
  low: {type: none}
anomaly_guard:
  enabled: true
  thresholds: ECC/outputs/thresholds/tensor_thresholds.json
channel:
  exclude_bf16_bits: [14]
  mode: ber            # ber | fixed
  ber: 1.0e-4
  seed: 42
output:
  root: outputs
```

所有"冻结值"集中在 `configs/v1_default.yaml` + `src/ma_uep/ecc/constants.py`，
并由 `test_frozen_config.py` 断言与任务书一致。

---

## 6. 迁移映射（旧 → 新）

| 现有文件 | 迁入 | 处理方式 |
|---|---|---|
| `ECC/allocation/build_level_map.py` | `src/ma_uep/uep/allocator.py` + `importance/ranker.py` | 拆分排序与分配 |
| `ECC/ecc/bch_high.py` | `src/ma_uep/ecc/bch_high.py` | 原样搬移 |
| `ECC/ecc/bch_medium.py` | `src/ma_uep/ecc/bch_medium.py` | 原样搬移 |
| `ECC/ecc/low_none.py` | `src/ma_uep/ecc/low_none.py` | 原样 |
| `ECC/ecc/level_codec.py` | `src/ma_uep/ecc/level_codec.py` | 原样 |
| `ECC/ecc/anomaly_guard.py` | `src/ma_uep/protection/anomaly_guard.py` | 原样 |
| `ECC/simulation/packed_codec.py` | `src/ma_uep/ecc/packed.py` | 原样 |
| `ECC/simulation/apply_ecc.py`（张量循环部分） | `src/ma_uep/pipeline/tensor_pass.py` | **合并** A2 |
| `ECC/simulation/fixed_errors.py`（同上） | 同上 | **合并** A2 |
| `ECC/evaluation/evaluate_ppl.py` | `src/ma_uep/evaluation/perplexity.py` + `pipeline/runner.py` | 拆分为评估与编排 |
| `ECC/evaluation/tensor_thresholds.py` | `scripts/generate_thresholds.py` | 保留 |
| `ECC/evaluation/wikitext_loader.py` | `src/ma_uep/model/dataset.py` | 加回退 |
| `ECC/evaluation/zero_weights.py` | `scripts/` 诊断脚本 | 保留 |
| `ECC/tests/*` | `tests/` | 改写为 pytest |

**原 `ECC/` 目录保持不动**，作为回归基准与回退。

---

## 7. 验证方案（如何证明"没改算法"）

| 层级 | 验证 | 判据 |
|---|---|---|
| L1 冻结常量 | `test_frozen_config.py` | 参数与任务书逐项相等 |
| L2 等级分配 | 重算 35 个 `level_map.npy` | 与现有产物**逐字节一致** |
| L3 码字 | 随机 10k 权重编码 | 与旧实现码字**逐位一致** |
| L4 功能 | High 1/2/3 bit、Medium 1 bit 穷举 | 全部正确纠正 |
| L5 无错闭环 | 35 张量编码→解码 | 改动权重 = 0 |
| L6 端到端 | 小子集（1 Block × 2 Tensor）跑完整流程 | 流程走通、结果落盘 |
| L7 历史回归 | seed=42、BER 1e-5/1e-4、固定 3-bit | ΔNLL 与历史差异 < 1e-3，差异需解释 |

L2/L3 是**逐位对比**，是最强的"未改算法"证据。

---

## 8. 实施计划（Phase 2–5）

| 阶段 | 内容 | 预估 |
|---|---|---|
| Phase 2 | 核心模块封装（model / importance / uep / ecc / channel / protection） | 2.5 h |
| Phase 3 | pipeline + experiment + CLI + 配置 | 1.5 h |
| Phase 4 | 测试（含 L2/L3 逐位对比）+ 端到端 + 历史回归 | 2 h |
| Phase 5 | README、架构文档、已知问题清单 | 1 h |
| **合计** | | **约 7 小时** |

---

## 9. 需要确认的三个决策

1. **数据不搬家**：16.34 GB 风险地图与 4.46 GB 重要性数组继续留在原位，
   新系统通过配置引用 + 校验和。→ 建议采纳（复制会再占 20 GB，磁盘不够）。
2. **基线改为实测**：取消硬编码 NLL，每次实验先跑一次 clean 评估（约 5 分钟）。
   → 建议采纳；历史常数保留在回归报告中说明。
3. **`ECC/` 原样保留**：不删除、不移动，作为回退与对比基准。
   → 建议采纳。
