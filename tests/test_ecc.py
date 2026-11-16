"""ECC 功能测试（标准库 unittest）。"""

import itertools
import unittest

import numpy as np

from ma_uep.ecc.level_codec import LevelCodec
from ma_uep.ecc.low_none import LowCodec
from ma_uep.ecc.packed import PackedCodec

DATA = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]


class TestCodecs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codecs = LevelCodec()

    def test_lengths(self):
        self.assertEqual(len(self.codecs.encode_weight(DATA, "high")), 31)
        self.assertEqual(len(self.codecs.encode_weight(DATA, "medium")), 24)
        self.assertEqual(len(self.codecs.encode_weight(DATA, "low")), 16)

    def test_clean_roundtrip(self):
        for level in ("high", "medium", "low"):
            codeword = self.codecs.encode_weight(DATA, level)
            self.assertEqual(
                self.codecs.decode_weight(codeword, level)["decoded_bits"],
                DATA, msg=level)

    def test_high_corrects_1_2_3(self):
        codeword = self.codecs.encode_weight(DATA, "high")
        for weight in (1, 2, 3):
            for positions in itertools.combinations(range(31), weight):
                received = list(codeword)
                for pos in positions:
                    received[pos] ^= 1
                result = self.codecs.decode_weight(received, "high")
                self.assertEqual(result["decoded_bits"], DATA,
                                 msg=f"w={weight} pos={positions}")

    def test_medium_corrects_1(self):
        codeword = self.codecs.encode_weight(DATA, "medium")
        for positions in itertools.combinations(range(24), 1):
            received = list(codeword)
            received[positions[0]] ^= 1
            result = self.codecs.decode_weight(received, "medium")
            self.assertEqual(result["decoded_bits"], DATA)

    def test_medium_never_overclaims(self):
        """2-bit 错误：数据若被改坏，必须标记 detected。"""
        codeword = self.codecs.encode_weight(DATA, "medium")
        rng = np.random.default_rng(2027)
        for _ in range(300):
            positions = rng.choice(24, size=2, replace=False)
            received = list(codeword)
            for pos in positions:
                received[int(pos)] ^= 1
            result = self.codecs.decode_weight(received, "medium")
            if result["decoded_bits"] != DATA:
                self.assertTrue(result["detected_error"])

    def test_low_has_no_parity(self):
        codec = LowCodec()
        self.assertEqual(codec.parity_bits, 0)
        self.assertEqual(codec.encode(DATA), DATA)

    def test_reject_bad_length(self):
        with self.assertRaises(ValueError):
            self.codecs.encode_weight([0] * 15, "high")
        with self.assertRaises(ValueError):
            self.codecs.decode_weight([0] * 30, "high")
        with self.assertRaises(ValueError):
            self.codecs.encode_weight(DATA, "ultra")


class TestPacked(unittest.TestCase):
    def test_packed_clean(self):
        rng = np.random.default_rng(7)
        values = rng.integers(0, 65536, size=2000, dtype=np.uint16)
        for level in ("high", "medium"):
            packed = PackedCodec(level)
            words = packed.encode(values)
            corrected, status = packed.decode(words)
            self.assertTrue((status == 0).all())
            self.assertTrue(np.array_equal(packed.data_of(corrected, status),
                                           values))

    def test_packed_single_error(self):
        rng = np.random.default_rng(11)
        values = rng.integers(0, 65536, size=500, dtype=np.uint16)
        for level, n_bits in (("high", 31), ("medium", 24)):
            packed = PackedCodec(level)
            words = packed.encode(values)
            corrupted = words.copy()
            pos = rng.integers(0, n_bits, size=len(words))
            for i, p in enumerate(pos):
                corrupted[i] ^= np.uint32(1 << int(p))
            corrected, status = packed.decode(corrupted)
            self.assertTrue(np.array_equal(packed.data_of(corrected, status),
                                           values), msg=level)


if __name__ == "__main__":
    unittest.main()
