"""实验配置：加载、校验与快照。"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..constants import (BLOCKS, DATASET_NAME, DATASET_SPLIT, EVAL_MAX_LENGTH,
                         EVAL_STRIDE, EXPERIMENT_MODES, LEVEL_RATIOS, MODULES,
                         SCHEMA_VERSION)


@dataclass
class ModelConfig:
    path: str = "models/Qwen2.5-1.5B-Base"
    device: str = "cuda:0"


@dataclass
class DatasetConfig:
    name: str = DATASET_NAME
    split: str = DATASET_SPLIT
    path: str | None = None
    stride: int = EVAL_STRIDE
    max_length: int = EVAL_MAX_LENGTH


@dataclass
class ScopeConfig:
    blocks: tuple[int, ...] = BLOCKS
    modules: tuple[str, ...] = MODULES


@dataclass
class ImportanceConfig:
    root: str = "task1_weight_importance/outputs/weight_importance"
    allocation_dir: str = "outputs/prepared/allocation"
    recompute: bool = False


@dataclass
class UEPConfig:
    ratios: dict = field(default_factory=lambda: dict(LEVEL_RATIOS))


@dataclass
class ECCConfig:
    bch_path: str | None = None


@dataclass
class GuardConfig:
    enabled: bool = True
    thresholds: str = "outputs/prepared/thresholds/tensor_thresholds.json"


@dataclass
class ChannelConfig:
    mode: str = "ber"
    ber: float = 1e-4
    bad_weights: int = 0
    k_bits: int = 1
    seed: int = 42


@dataclass
class OutputConfig:
    root: str = "MA-UEP/outputs"


@dataclass
class ExperimentConfig:
    schema_version: str = SCHEMA_VERSION
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    uep: UEPConfig = field(default_factory=UEPConfig)
    ecc: ECCConfig = field(default_factory=ECCConfig)
    anomaly_guard: GuardConfig = field(default_factory=GuardConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mode: str = "ma_uep"
    max_windows: int | None = None

    # ---------------- 校验 ----------------
    def validate(self) -> None:
        if self.mode not in EXPERIMENT_MODES:
            raise ValueError(f"未知实验模式 {self.mode!r}，可选 {EXPERIMENT_MODES}")
        if self.channel.mode not in ("ber", "fixed"):
            raise ValueError("channel.mode 必须是 ber 或 fixed")
        if self.channel.mode == "ber" and self.channel.ber < 0:
            raise ValueError("ber 必须非负")
        if abs(sum(self.uep.ratios.values()) - 1.0) > 1e-9:
            raise ValueError("UEP 比例之和必须为 1")
        if tuple(sorted(self.uep.ratios)) != ("high", "low", "medium"):
            raise ValueError("UEP 比例必须包含 high/medium/low")

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def snapshot(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    # ---------------- 加载 ----------------
    @classmethod
    def from_dict(cls, payload: dict) -> "ExperimentConfig":
        payload = dict(payload or {})

        def pick(key, klass):
            data = payload.get(key) or {}
            valid = {f.name for f in dataclasses.fields(klass)}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(f"配置节 {key} 含未知字段: {sorted(unknown)}")
            return klass(**data)

        cfg = cls(
            model=pick("model", ModelConfig),
            dataset=pick("dataset", DatasetConfig),
            scope=pick("scope", ScopeConfig),
            importance=pick("importance", ImportanceConfig),
            uep=pick("uep", UEPConfig),
            ecc=pick("ecc", ECCConfig),
            anomaly_guard=pick("anomaly_guard", GuardConfig),
            channel=pick("channel", ChannelConfig),
            output=pick("output", OutputConfig),
            mode=payload.get("mode", "ma_uep"),
            max_windows=payload.get("max_windows"),
        )
        if "blocks" in payload.get("scope", {}):
            cfg.scope.blocks = tuple(payload["scope"]["blocks"])
        if "modules" in payload.get("scope", {}):
            cfg.scope.modules = tuple(payload["scope"]["modules"])
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        import yaml
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
