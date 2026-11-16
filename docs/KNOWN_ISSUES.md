# 已知问题与后续改进清单

> 对应审计报告中的问题编号（A/B/C）。

## 一、已解决

| 编号 | 问题 | 解决方式 |
|---|---|---|
| A1 | 层次倒置（evaluation 导入 simulation） | 拆为 `pipeline/runner.py`（编排）+ `evaluation/perplexity.py`（纯评估） |
| A2 | `apply_ecc` 与 `fixed_errors` 重复实现 | 合并为唯一 `pipeline/tensor_pass.py`，信道通过策略注入 |
| A3 | 无统一入口 | `run.py` + `ma_uep/cli.py`（config/check/run/batch/report/test） |
| A4 | 无配置层 | `configs/v1_default.yaml` + `ExperimentConfig` 校验 |
| A5 | 无实验 ID 与元数据 | `ExperimentRegistry`，每次实验生成 `<时间>-<模式>-<参数>-<随机后缀>` |
| A6 | 12 处 `sys.path.insert` | 收敛为 `ecc/vendor.py` 一处（可配置 `MA_UEP_BCH_PATH`） |
| B1 | 两个基线常数并存 | **取消所有硬编码**，基线由每次实验实测写入 `meta.json` |
| B2 | `fixed_errors` 缺 `anomaly_guard` 字段 | 统一结果 schema，字段由 `result.json` 保证 |
| B3 | `run_experiments` 不支持 guard | `--guard true/false` 统一入口 |
| B4 | 数据集写死 HF 缓存路径 | `model/dataset.py` 三级回退 + 可配置 `dataset.path` |
| C3 | 未记录 Git / 环境 | `meta.json` 记录 commit、Python、torch、transformers、CUDA 设备 |
| C4 | 就地修改权重无备份 | 改为"磁盘 checkpoint 从不写入 + 指纹记录"，避免整模型内存备份 |

## 二、未解决 / 需要注意

| 编号 | 问题 | 影响 | 计划 |
|---|---|---|---|
| **N1** | **分块信道改变随机流** | 新系统与旧 `ECC/` 脚本不是逐样本同随机流；同一 seed 下注入错误总数相差约 0.03%，ΔNLL 差异在 1e-3 量级 | 若要严格逐位复现，需要"整组抽样 + 分块应用"的两段式信道（见下） |
| N2 | 只支持 5 个 Block / 35 个 Tensor | 不是全模型覆盖（任务书已明确） | 全模型需要重新生成风险地图（16 GB 级） |
| N3 | `batch` 模式逐个 JSON 重建模型 | 每次实验重新加载模型（约 20 s），GPU 内存反复申请 | 可加"模型复用"开关，但要注意权重状态隔离 |
| N4 | 无硬件指标 | 只有 Python 耗时 | 任务书明确不做 RTL |
| N5 | Medium 仍为 d=3 | 3-bit 错误可能静默误纠正 | 已由 Anomaly Guard 兜底；如需根治需提高码距 |

## 三、N1 的技术说明（已解决，保留供追溯）

旧实现：每个 (Tensor, 等级) 一次 `Poisson` 抽样 + 一次 `rng.choice`，然后整组注入。
新实现：按 100 万权重分块，每块独立抽样。

- **统计等价**：错误率一致，期望总错误数一致。
- **非逐位等价**：同一 seed 下具体哪些权重出错不同。

实测（BER=1e-4, seed=42, 20 窗口）：

| 指标 | 旧实现 | 新实现 |
|---|---:|---:|
| 注入错误 | 397,234 | 397,129 |
| 改动权重 | 280,814 | 280,797 |

差异 0.03%，属于分块抽样的正常波动。

**如果后续要求严格复现旧脚本**，可以让 `TensorProcessor` 接收一个 `whole_group=True`
开关，对该 Tensor 一次性抽取位置、再分块应用（内存 O(n_err)，通常 < 5 MB）。

## 四、性能记录

| 操作 | 耗时 |
|---|---|
| 35 个 Tensor 的编码 + 注错 + 译码 | 约 90 s |
| 一次完整模型评估（2335 窗口） | 约 150 s |
| 一次完整实验（含基线，max_windows=20） | 约 60 s |

## 五、环境约束

- 本机可用提交内存波动大（曾低至 30 MB），因此所有大数组操作必须分块；
- 磁盘 C: 长期紧张，所有输出写 D:；
- GPU 8 GB，模型 bf16 约 3.1 GB。
