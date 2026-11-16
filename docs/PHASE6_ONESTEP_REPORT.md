# 一条龙方案实施报告

> 完成时间：2026-11-16
> 结论：**代码与测试全部完成**；`prepare` 的真实全量运行被本机资源阻塞（C 盘 0.75 GB，
> 页面文件无法扩展，CUDA 上下文耗尽），已用真权重切片验证核心算法。

---

## 1. 已实现

| 组件 | 文件 | 说明 |
|---|---|---|
| 项目根唯一定义 | `src/ma_uep/paths.py` | 消除 6 处 `parents[3]/[4]` 硬编码 |
| F 采集 | `importance/activation.py` | forward_pre_hook 累加各通道平方和，float64 |
| Smax 扫描 | `importance/scan.py` | 分块流式，只保留 `max_b`，不落 16 GB 地图 |
| 一条龙编排 | `importance/prepare.py` | F → Smax → 排序 → 等级图 + 阈值 |
| CLI | `cli.py` `prepare` 子命令 | `--all-blocks` / `--keep-smax` / `--save-risk-map` |
| BCH 打包 | `ecc/_vendor/BCH/` | 4 个源文件，随仓库分发 |
| 配置默认值 | `config.py` / `v1_default.yaml` | 指向 `outputs/prepared/` |
| 数据集口径 | `model/dataset.py` | 改为 `\n\n` 拼接，token 数与历史一致（299,078） |

## 2. 验证结果

| 验证 | 方法 | 结果 |
|---|---|---|
| 单元测试 | `python -m unittest discover` | **41 项通过** |
| Smax 公式 | 合成数据 vs 暴力逐位重算 | 完全一致（rtol 1e-5） |
| 分级逻辑 | 合成 Smax，检查 5% / 15% | 通过 |
| **真权重 Smax** | L0.k_proj 前 5000 权重 vs 历史 `I_max_log10` | **log10 最大差 1.16e-7**（相对 3.5e-8） |
| BCH 打包 | 从 vendored 目录构造 BCH(31,16,7) | n=31/k=16/t=3 ✓ |

真权重那项是最关键的：说明**扫描公式、F 口径、位序全部正确**，
只是没能在整模型规模上跑完。

## 3. 被阻塞的部分

| 项 | 阻塞原因 |
|---|---|
| `run.py prepare` 全量运行（35 个 Tensor） | 模型加载需约 3 GB 连续提交内存；本机 C 盘仅剩 0.75 GB，页面文件无法扩展，报 `OSError 1455` |
| 真权重多 Tensor 扫描脚本 | 在内存压力下进程崩溃（0xC0000005） |

尝试过：`device_map` 直载 GPU（CUDA OOM）、清理缓存（仅回收 0.6 GB）、
缩小分块（仍崩溃）。

## 4. 恢复条件与验证命令

当 C 盘有 ≥ 5 GB 空闲、且没有其他 GPU 进程时，执行：

```bash
cd MA-UEP
python run.py prepare --keep-smax
```

预期约 12 分钟，产物在 `outputs/prepared/`。

随后验证与历史等级图的一致性：

```bash
python - <<'PY'
import numpy as np, json
from pathlib import Path
hi, med = 9.404956e-01, 4.150179e-01   # 历史全局阈值
new_root = Path("outputs/prepared/allocation")
old_root = Path("../ECC/outputs/allocation")
total = agree = 0
for d in sorted(old_root.glob("L*_*")):
    n = np.load(new_root/d.name/"level_map.npy")
    o = np.load(d/"level_map.npy")
    total += o.size; agree += int((n==o).sum())
print("分类一致率: %.4f%%" % (100*agree/total))
PY
```

预期一致率 > 99.99%（阈值边界可能有极少数抖动）。

## 5. 发布物变化

| 项 | 原方案 | 现方案 |
|---|---|---|
| 仓库体积 | 33 MB（含等级图） | **约 2 MB**（不含任何大数据） |
| 等级图来源 | 预置文件 | **运行时 `prepare` 生成** |
| BCH 母码 | 依赖外部目录 | **随仓库打包** |
| 首次使用 | 直接跑 | 先跑一次 `prepare`（约 12 分钟） |

GitHub Release 仍可附上参考等级图（31.5 MB），供想跳过准备的人使用。
