"""冻结参数与任务书一致性测试（标准库 unittest，无需 pytest）。"""

import unittest

from ma_uep import constants as C
from ma_uep.experiment.config import ExperimentConfig


class TestFrozenConstants(unittest.TestCase):
    def test_version(self):
        self.assertEqual(C.SYSTEM_VERSION, "MA-UEP V1.0")
        self.assertEqual(C.SCHEMA_VERSION, "1.0")

    def test_scope(self):
        self.assertEqual(C.BLOCKS, (0, 2, 15, 23, 27))
        self.assertEqual(len(C.MODULES), 7)
        self.assertIn("NOT a full", C.COVERAGE_NOTE)

    def test_bit_pool(self):
        self.assertEqual(C.BF16_BITS, 16)
        self.assertEqual(C.EXCLUDED_BF16_BITS, (14,))
        self.assertEqual(len(C.RISK_BITS), 15)
        self.assertNotIn(14, C.RISK_BITS)

    def test_level_ratios(self):
        self.assertEqual(C.LEVEL_RATIOS,
                         {"high": 0.05, "medium": 0.15, "low": 0.80})
        self.assertAlmostEqual(sum(C.LEVEL_RATIOS.values()), 1.0, places=12)

    def test_bch_high_frozen(self):
        self.assertEqual(C.BCH_HIGH["m"], 5)
        self.assertEqual(C.BCH_HIGH["primitive_poly"], [1, 0, 1, 0, 0, 1])
        self.assertEqual(C.BCH_HIGH["t"], 3)
        self.assertEqual((C.BCH_HIGH["n"], C.BCH_HIGH["k"]), (31, 16))
        self.assertEqual(C.BCH_HIGH["parity_bits"], 15)
        self.assertEqual(C.BCH_HIGH["guaranteed_correction"], 3)

    def test_bch_medium_frozen(self):
        self.assertEqual(C.BCH_MEDIUM["keep_parity"],
                         (0, 2, 4, 6, 8, 10, 12, 14))
        self.assertEqual((C.BCH_MEDIUM["n"], C.BCH_MEDIUM["k"]), (24, 16))
        self.assertEqual(C.BCH_MEDIUM["parity_bits"], 8)
        self.assertEqual(C.BCH_MEDIUM["distance"], 3)
        self.assertEqual(C.BCH_MEDIUM["guaranteed_correction"], 1)

    def test_channel_excludes_b14(self):
        self.assertEqual(C.CHANNEL_EXCLUDED_BF16_BITS, (14,))

    def test_modes(self):
        self.assertTrue({"baseline", "no_protection", "ma_uep",
                         "random_uep", "ma_uep_without_guard"}
                        <= set(C.EXPERIMENT_MODES))


class TestConfig(unittest.TestCase):
    def test_default_config_validates(self):
        cfg = ExperimentConfig()
        cfg.validate()
        self.assertEqual(cfg.mode, "ma_uep")

    def test_reject_bad_mode(self):
        cfg = ExperimentConfig()
        cfg.mode = "nope"
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_reject_bad_ratios(self):
        cfg = ExperimentConfig()
        cfg.uep.ratios = {"high": 0.5, "medium": 0.5, "low": 0.5}
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_reject_unknown_field(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({"model": {"unknown": 1}})

    def test_roundtrip_dict(self):
        cfg = ExperimentConfig()
        payload = cfg.to_dict()
        cfg2 = ExperimentConfig.from_dict(payload)
        self.assertEqual(cfg2.mode, cfg.mode)
        self.assertEqual(cfg2.channel.ber, cfg.channel.ber)


if __name__ == "__main__":
    unittest.main()
