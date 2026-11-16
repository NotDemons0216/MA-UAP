"""L2/L3 等价性验证：证明重构没有改变冻结算法。

L2：重新计算 35 个 level_map，与 ECC/outputs/allocation 下的现有产物逐字节比较。
L3：随机 10k 个权重，比较新旧 PackedCodec 的码字与译码结果逐位一致。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "MA-UEP" / "src"))
sys.path.insert(0, str(REPO / "ECC" / "simulation"))
sys.path.insert(0, str(REPO / "ECC" / "ecc"))

from ma_uep.importance.source import ImportanceSource          # noqa: E402
from ma_uep.importance.ranker import GlobalRanker              # noqa: E402
from ma_uep.ecc.packed import PackedCodec as NewPacked         # noqa: E402

TASK1 = REPO / "task1_weight_importance" / "outputs" / "weight_importance"
OLD_ALLOC = REPO / "ECC" / "outputs" / "allocation"
SCRATCH = REPO / "MA-UEP" / "outputs" / "equivalence"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_l2() -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source = ImportanceSource(TASK1)
    ranker = GlobalRanker(source)
    thresholds = ranker.thresholds()
    ranker.write_level_maps(thresholds, SCRATCH)

    results = {"thresholds": thresholds, "tensors": {}, "all_match": True}
    for name in source.tensors:
        key = name.replace(".", "_")          # L0.q_proj -> L0_q_proj
        new_path = SCRATCH / key / "level_map.npy"
        old_path = OLD_ALLOC / key / "level_map.npy"
        if not old_path.exists():
            results["tensors"][name] = {"status": "old_missing"}
            results["all_match"] = False
            continue
        new_hash = sha256_file(new_path)
        old_hash = sha256_file(old_path)
        same = new_hash == old_hash
        results["tensors"][name] = {"same_bytes": same,
                                    "new_sha256": new_hash[:16],
                                    "old_sha256": old_hash[:16]}
        results["all_match"] &= same
    return results


def check_l3(n: int = 10000, seed: int = 2027) -> dict:
    """新旧 PackedCodec 逐位比较。"""
    from packed_codec import PackedCodec as OldPacked  # 旧实现

    rng = np.random.default_rng(seed)
    values = rng.integers(0, 65536, size=n, dtype=np.uint16)
    out = {}
    for level in ("high", "medium"):
        new = NewPacked(level)
        old = OldPacked(level)
        new_words = new.encode(values)
        old_words = old.encode(values)
        same_encode = bool(np.array_equal(new_words, old_words))

        # 注入 1~3 bit 错误后比较译码
        rng2 = np.random.default_rng(seed + 1)
        corrupted = new_words.copy()
        for k_bits in (1, 2, 3):
            bad = new_words.copy()
            for i in range(len(bad)):
                pos = rng2.choice(new.n, size=k_bits, replace=False)
                for p in pos:
                    bad[i] ^= np.uint32(1 << int(p))
            new_corr, new_status = new.decode(bad)
            old_corr, old_status = old.decode(bad)
            same_decode = bool(np.array_equal(new_corr, old_corr)
                               and np.array_equal(new_status, old_status))
            out[f"{level}_k{k_bits}"] = {"same": same_decode}
            del bad, new_corr, old_corr
        out[f"{level}_encode"] = {"same": same_encode,
                                  "n": int(n)}
    return out


def main() -> int:
    print("=== L2：重新计算等级图并与现有产物逐字节比较 ===")
    l2 = check_l2()
    print(f"全部一致: {l2['all_match']}")
    mismatched = [k for k, v in l2["tensors"].items() if not v.get("same_bytes")]
    if mismatched:
        print(f"不一致: {mismatched[:10]}（共 {len(mismatched)}）")
    print(f"阈值: high={l2['thresholds']['high_threshold_smax']:.6e} "
          f"medium={l2['thresholds']['medium_boundary_threshold_smax']:.6e}")

    print("\n=== L3：新旧 PackedCodec 逐位比较 ===")
    l3 = check_l3()
    all_ok = True
    for key, item in l3.items():
        all_ok &= bool(item["same"])
        print(f"  {key:16s} same={item['same']}")

    print(f"\nL2={'PASS' if l2['all_match'] else 'FAIL'}  "
          f"L3={'PASS' if all_ok else 'FAIL'}")
    return 0 if (l2["all_match"] and all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
