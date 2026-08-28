"""Tests for transforms (logit/expit/microprice/clip/round)."""

import math
import unittest

import numpy as np

from polymarket_as.transforms import (
    clip_price, expit, logit, microprice, price_space_sigma, round_to_tick,
)


class TestLogit(unittest.TestCase):
    def test_roundtrip(self):
        for p in [0.05, 0.1, 0.5, 0.9, 0.95]:
            self.assertAlmostEqual(float(expit(logit(p))), p, places=10)

    def test_logit_at_half_is_zero(self):
        self.assertAlmostEqual(float(logit(0.5)), 0.0, places=12)

    def test_clipping_handles_zero_and_one(self):
        # No exception, finite output
        self.assertTrue(math.isfinite(float(logit(0.0))))
        self.assertTrue(math.isfinite(float(logit(1.0))))


class TestMicroprice(unittest.TestCase):
    def test_balanced_book(self):
        # Equal queue → microprice should be at the mid
        self.assertAlmostEqual(microprice(0.5, 0.52, 100, 100), 0.51, places=10)

    def test_imbalanced_book_pushes_toward_thin_side(self):
        # Lots of bid depth, thin ask → microprice closer to ask (taker is likely to lift offer)
        mp = microprice(best_bid=0.5, best_ask=0.52, bid_qty=1000, ask_qty=100)
        self.assertGreater(mp, 0.51)
        self.assertLess(mp, 0.52)

    def test_empty_returns_none(self):
        self.assertIsNone(microprice(0.5, 0.52, 0, 0))


class TestPriceSpaceSigma(unittest.TestCase):
    def test_collapses_at_boundary(self):
        # σ_S = σ_logit · S(1−S) → 0 at S=0 or S=1
        self.assertAlmostEqual(price_space_sigma(0.1, 0.0), 0.0)
        self.assertAlmostEqual(price_space_sigma(0.1, 1.0), 0.0)

    def test_max_at_half(self):
        # peak of S(1−S) is at 0.5 with value 0.25
        self.assertAlmostEqual(price_space_sigma(0.1, 0.5), 0.025, places=10)


class TestRoundAndClip(unittest.TestCase):
    def test_round_to_tick(self):
        self.assertAlmostEqual(round_to_tick(0.5234, 0.01), 0.52, places=10)
        self.assertAlmostEqual(round_to_tick(0.5251, 0.01), 0.53, places=10)
        self.assertAlmostEqual(round_to_tick(0.5249, 0.001), 0.525, places=10)

    def test_clip_price(self):
        self.assertEqual(clip_price(-0.1, 0.01), 0.01)
        self.assertEqual(clip_price(1.5, 0.01), 0.99)
        self.assertEqual(clip_price(0.5, 0.01), 0.5)


if __name__ == "__main__":
    unittest.main()
