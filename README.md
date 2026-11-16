# MA-UEP V1.0

**Model-Aware Unequal Error Protection for LLM weights.**

根据"每个权重最危险的 bit 会造成多大扰动"，把权重分成三档，给不同强度的 BCH 纠错保护，
并用真实模型验证效果。

```
权重重要性分析 → UEP 资源分配 → BCH 编码 → 信道注错
     → BCH 译码 → 异常遏制 → 模型恢复 → NLL / PPL 评估
```

| 等级 | 全局排名 | 占比 | 编码 | 保证纠错 |
|---|---|---:|---|---|
| High | 前 5% | 5% | BCH(31,16,7) | 3 bit |
| Medium | 接下来 15% | 15% | BCH(24,16) 删余码 | 1 bit |
| Low | 其余 | 80% | 无 | — |

整体码率约 **89.1%**。

> **覆盖范围**：默认处理 Qwen2.5-1.5B-Base 的 5 个代表 Block（L0/L2/L15/L23/L27）
> × 7 个 Linear = 35 个 Tensor、233,963,520 个权重。
> 这是代表性范围，**不是全模型 28 个 Block**。

---

## 安装

```bash
git clone https://github.com/NotDemons61/MA-UEP.git
cd MA-UEP
pip install -e .
```

依赖：Python ≥ 3.10、numpy、pandas、pyyaml、torch、transformers、pyarrow、safetensors。

BCH 母码已随仓库附带（`src/ma_uep/ecc/_vendor/BCH/`），无需额外下载。

---

## 准备模型

系统不自带模型权重，请自行下载：

```bash
# 方式一：huggingface-cli
huggingface-cli download Qwen/Qwen2.5-1.5B-Base --local-dir models/Qwen2.5-1.5B-Base

# 方式二：任意方式下载后，把路径写进 configs/v1_default.yaml 的 model.path
```

---

## 一条龙准备等级图（首次约 12 分钟）

```bash
python run.py prepare
```

它会依次完成：

| 步骤 | 内容 | 耗时 |
|---|---|---:|
| ① | 在 WikiText-2 上采集 35 个 Linear 的输入激活 RMS（F） | ~2 分钟 |
| ② | 逐权重逐 bit 计算 ΔW，得到 Smax = max_b F·|ΔW| | ~8 分钟 |
| ③ | 全局排序 → 高/中/低等级图 + 异常遏制阈值 | ~1 分钟 |

产物：

```text
outputs/prepared/
├── allocation/          # 等级图（.npy，每 Tensor 一个）
├── thresholds/          # 异常遏制阈值
└── prepare_meta.json    # 模型路径、数据来源、耗时、等级统计
```

**可选参数**：

| 参数 | 作用 |
|---|---|
| `--all-blocks` | 处理全部 28 个 Block（约 60 分钟） |
| `--keep-smax` | 保留每权重 Smax（约 936 MB） |
| `--save-risk-map` | 额外保存 Bit-level 风险地图（约 16 GB） |

> 数据分层：Bit 风险地图（16 GB）→ Smax（936 MB）→ 等级图（234 MB）→ 阈值（6 KB）。
> 默认只保留最后两层；前两层是研究用的中间产物，运行时不需要。

---

## 跑实验

```bash
python run.py check                                    # 检查环境与数据
python run.py run --mode ma_uep --ber 1e-4 --seed 42   # 主方案
python run.py run --mode no_protection --ber 1e-4      # 无保护对照
python run.py run --mode random_uep --ber 1e-4         # 随机分配对照
python run.py run --mode ma_uep_without_guard --ber 1e-4   # 关闭异常遏制
python run.py run --mode ma_uep --ber 1e-4 --max-windows 50  # 快速试跑
```

### 实验模式

| 模式 | ECC | 异常遏制 | 用途 |
|---|---|---|---|
| `baseline` | ✗ | ✗ | 原始模型 NLL |
| `no_protection` | ✗ | ✗ | 故障基线 |
| `ma_uep` | ✓ | ✓ | 主方案 |
| `ma_uep_without_guard` | ✓ | ✗ | 单独评估遏制贡献 |
| `random_uep` | ✓ | ✓ | 验证重要性排序的价值 |

### 批量

```bash
python run.py batch --config configs/experiment_grid.yaml
```

---

## 结果

每次实验生成独立目录：

```text
outputs/<实验ID>/
├── meta.json            # 实验 ID、Git commit、环境、配置
├── config.snapshot.json
├── result.json          # 完整结果（逐 Tensor 明细）
├── summary.csv
└── REPORT.md
```

关键字段：`baseline.nll`、`evaluation.nll`、`evaluation.delta_nll`、
`uep.{high,medium,low,code_rate}`、`channel.total.*`、`tensors[*].anomaly_guard.*`。

---

## 测试

```bash
python run.py test                          # 41 项单元测试
python tests/verify_equivalence.py          # 与历史产物逐字节/逐位比对（需旧产物）
python tests/run_regression.py              # 历史实验回归（需模型）
```

---

## 配置

`configs/v1_default.yaml`，命令行可覆盖。**冻结参数**（BCH 参数、删余位置、
等级比例、b14 排除）在 `src/ma_uep/constants.py`，由 `tests/test_frozen_config.py` 锁定。

---

## 目录结构

```text
MA-UEP/
├── run.py
├── configs/v1_default.yaml
├── src/ma_uep/
│   ├── paths.py            # 项目根唯一定义
│   ├── constants.py        # 冻结参数
│   ├── model/              # 加载、BF16 检查、数据集、评估
│   ├── importance/         # F 采集、Smax 扫描、全局排序、prepare
│   ├── uep/                # 等级分配与码率统计
│   ├── ecc/                # BCH High/Medium/Low + 打包向量化 + _vendor/BCH
│   ├── channel/            # 错误注入
│   ├── protection/         # 异常遏制
│   ├── pipeline/           # 唯一张量遍历 + 12 步流程
│   ├── evaluation/         # NLL/PPL
│   └── experiment/         # 配置、注册表、报告
├── tests/
├── docs/
└── outputs/
```

---

## 限制

1. 默认只覆盖 5 个 Block / 35 个 Tensor，**不是全模型**；
2. Medium 为 d=3，3-bit 错误可能静默误纠正（由异常遏制兜底）；
3. 异常遏制只抓幅值异常，抓不到中等幅值的错误；
4. 无 RTL / 硬件评估；
5. 详见 `docs/KNOWN_ISSUES.md`。

---

## 文档

| 文件 | 内容 |
|---|---|
| `docs/PHASE0_AUDIT.md` | 代码审计 |
| `docs/PHASE1_ARCHITECTURE.md` | 架构设计 |
| `docs/PHASE2_3_REPORT.md` | 封装进度 |
| `docs/PHASE4_5_REPORT.md` | 测试与回归 |
| `docs/KNOWN_ISSUES.md` | 已知问题 |
| `docs/PUBLISH_PLAN.md` | 发布方案 |

---

## 引用

如果你在研究中使用了本系统，请引用本仓库并注明 MA-UEP V1.0。
