"""MA-UEP V1.0 冻结参数。

本文件是"冻结算法"的唯一真实来源。任何修改都必须先更新任务书，
并由 tests/test_frozen_config.py 断言。
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0"
SYSTEM_VERSION = "MA-UEP V1.0"

# ---------------- 模型与范围 ----------------
MODEL_NAME = "Qwen2.5-1.5B-Base"
MODEL_DTYPE = "bfloat16"
DATASET_NAME = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
EVAL_STRIDE = 128
EVAL_MAX_LENGTH = 256

BLOCKS = (0, 2, 15, 23, 27)
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
COVERAGE_NOTE = ("35 tensors of 5 representative blocks; "
                 "NOT a full 28-block model coverage")

# ---------------- 重要性 ----------------
BF16_BITS = 16
EXCLUDED_BF16_BITS = (14,)                      # 冻结：b14 不进入普通风险统计
RISK_BITS = tuple(b for b in range(BF16_BITS) if b not in EXCLUDED_BF16_BITS)  # 15 位
IMPORTANCE_METRIC = "smax"                       # Smax(w) = max_b S(w,b)
RANKING_SCOPE = "global"                         # 跨全部 Tensor 全局排序

# ---------------- UEP 等级（冻结比例）----------------
LEVEL_RATIOS = {"high": 0.05, "medium": 0.15, "low": 0.80}
LEVEL_ORDER = ("high", "medium", "low")
LEVEL_CODES = {"low": 0, "medium": 1, "high": 2}

# ---------------- ECC（冻结码型）----------------
BCH_HIGH = {
    "name": "bch_31_16_7",
    "m": 5,
    "primitive_poly": [1, 0, 1, 0, 0, 1],       # x^5 + x^2 + 1
    "t": 3,
    "n": 31,
    "k": 16,
    "parity_bits": 15,
    "guaranteed_correction": 3,
}

BCH_MEDIUM = {
    "name": "pruned_bch_24_16",
    "n": 24,
    "k": 16,
    "parity_bits": 8,
    "keep_parity": (0, 2, 4, 6, 8, 10, 12, 14),  # 冻结删余位置
    "distance": 3,
    "guaranteed_correction": 1,
}

LOW_NONE = {"name": "none", "n": 16, "k": 16, "parity_bits": 0,
            "guaranteed_correction": 0}

# ---------------- 信道 ----------------
CHANNEL_EXCLUDED_BF16_BITS = (14,)   # 信道同样排除 b14（RF01 口径）
CHANNEL_MODES = ("ber", "fixed")

# ---------------- Anomaly Guard ----------------
GUARD_RULE = "nonfinite(w) OR |w| > T_k"
GUARD_ACTION = "set to zero"
GUARD_THRESHOLD_SOURCE = "per-tensor max |w| of the clean checkpoint"

# ---------------- 实验模式 ----------------
EXPERIMENT_MODES = ("baseline", "no_protection", "ma_uep", "random_uep",
                    "ma_uep_without_guard")
