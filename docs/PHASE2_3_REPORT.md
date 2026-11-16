# Phase 2–3 完成报告：核心模块封装与完整流水线

> 完成时间：2026-11-14
> 状态：**核心模块 + 统一入口 + 端到端跑通**；Phase 4 全量回归与 Phase 5 文档未完成

---

## 1. 已完成内容

### 1.1 目录结构

```text
MA-UEP/
├── run.py                       # 统一入口
├── pyproject.toml
├── configs/v1_default.yaml      # 冻结配置
├── src/ma_uep/
│   ├── constants.py             # 冻结参数唯一来源
│   ├── cli.py                   # config/check/run/batch/report/test
│   ├── model/{loader,dataset}.py
│   ├── importance/{source,ranker}.py
│   ├── uep/allocator.py
│   ├── ecc/{vendor,bch_high,bch_medium,low_none,level_codec,packed}.py
│   ├── channel/injection.py
│   ├── protection/anomaly_guard.py
│   ├── pipeline/{tensor_pass,runner}.py
│   ├── evaluation/perplexity.py
│   └── experiment/{config,registry,reporter}.py
├── tests/                       # 37 项单元测试
└── docs/{PHASE0_AUDIT,PHASE1_ARCHITECTURE,KNOWN_ISSUES}.md
```

### 1.2 已解决的审计问题

A1–A6、B1–B4、C3、C4 全部处理（详见 `KNOWN_ISSUES.md` §1）。

### 1.3 验证结果

| 层级 | 内容 | 结果 |
|---|---|---|
| 单元测试 | 37 项（冻结参数 / ECC / 分配 / 守护 / 信道 / 配置） | **37 passed** |
| **L2** | 重算 35 个等级图，与冻结产物逐字节比对 | **SHA-256 全部一致** |
| **L3** | 新旧 PackedCodec 编码 + 1/2/3-bit 译码 | **逐位一致** |
| L6 端到端 | `run.py run --mode ma_uep --ber 1e-4`（20 窗口） | **跑通并落盘** |

**L2/L3 证明重构没有改变冻结算法。**

### 1.4 端到端产物（示例实验）

```text
outputs/20261114-183808-ma_uep-ber0p0001-seed42-99cd92/
├── meta.json          # 实验 ID、Git、环境、配置
├── config.snapshot.json
├── result.json        # 完整结果（50 KB）
├── summary.csv
└── REPORT.md
```

关键字段：

| 项 | 值 |
|---|---|
| High / Medium / Low | 11,797,359 / 35,150,400 / 187,015,761 |
| 码率 | 0.890954 |
| 注入错误 | 397,129 |
| 已纠正 | 116,003 |
| 不可纠正 | 85 |
| 改动权重 | 280,797 |
| ΔNLL（20 窗口） | +0.00746 |

### 1.5 与旧实现的对照

| 指标 | 旧脚本 | 新系统 | 差异 |
|---|---:|---:|---:|
| 注入错误 | 397,234 | 397,129 | 0.03% |
| 改动权重 | 280,814 | 280,797 | 0.006% |

差异来自分块抽样（见 `KNOWN_ISSUES.md` N1），统计等价。

---

## 2. 修改的文件

**新建**（全部在 `MA-UEP/`）：20 个源码文件、6 个测试文件、2 个配置文件、3 个文档。
**未修改**：`ECC/`、`BCH/`、`Hamming/`、`task1_*`、`v06x/` 以及全部历史结果。

---

## 3. 尚未解决的问题

| 编号 | 问题 |
|---|---|
| N1 | 分块信道导致与旧脚本非逐样本同随机流（统计等价，见文档） |
| N2 | 仅覆盖 35 个 Tensor |
| N3 | batch 模式每次重建模型（约 20 s/次） |
| N5 | Medium 仍为 d=3，靠 Guard 兜底 |

---

## 4. 下一阶段计划（Phase 4–5）

### Phase 4：测试与历史回归（约 2 h）

1. 无错误闭环：全 35 Tensor 编码→解码，改动权重 = 0；
2. 端到端：baseline / no_protection / ma_uep / random_uep / ma_uep_without_guard；
3. **历史回归**：BER 1e-5、1e-4、固定 3-bit × 3 seeds，与冻结表比对；
4. 记录差异并解释（预期在 1e-3 ~ 1e-2 量级，源于 N1）。

### Phase 5：文档与发布（约 1 h）

1. `README.md`（安装、配置、数据准备、单次/批量实验、结果查看）；
2. 架构说明；
3. 验收清单勾选。

---

## 5. 需要确认

Phase 4 的历史回归会暴露 N1 带来的差异。有两种处理方式：

- **方案 A（默认）**：接受统计等价，把差异记录在回归报告中；
- **方案 B**：实现"整组抽样 + 分块应用"的信道，让新系统与旧脚本**逐样本可复现**
  （约多 1 小时，且需要为 1376 万权重保存位置数组，内存峰值约 n_err×8 B ≈ 3 MB）。

建议选 **B**，因为"可复现"是本任务的核心要求之一。
