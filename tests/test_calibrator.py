"""Validate that the calibrator recovers ground-truth parameters from synthetic streams.

Tolerances are set per ENG_NOTE.md §9: σ ±25%, κ ±30%, A ±50% over a 5-minute
simulated stream. We use longer sims here for tighter checks.
"""

import unittest

import numpy as np

from polymarket_as._simulator import SyntheticParams, simulate
from polymarket_as.calibrator import ASCalibrator
from polymarket_as.state import BookSnapshot, OrderBookState, TradeEvent


def feed(state: OrderBookState, events):
    for e in events:
        if isinstance(e, BookSnapshot):
            state.ingest_snapshot(e)
        else:
            state.ingest_trade(e)


class TestCalibratorOnSynthetic(unittest.TestCase):
    def _run(self, params: SyntheticParams):
        events = simulate(params)
        # Window bigger than the sim duration → calibrator sees everything
        state = OrderBookState(window_seconds=params.duration_seconds + 10.0)
        feed(state, events)
        calib = ASCalibrator()
        result = calib.fit(state)
        self.assertIsNotNone(result, msg="calibrator failed to produce a fit")
        self.assertTrue(result.is_valid())
        return result

    def test_recovers_sigma(self):
        # Long simulation for a tight sigma fit.
        params = SyntheticParams(
            duration_seconds=3600.0, sigma_logit=0.05,
            A_buy=1.0, A_sell=1.0, kappa_buy=50.0, kappa_sell=50.0,
            seed=42,
        )
        result = self._run(params)
        rel_err_sigma = abs(result.sigma_logit - params.sigma_logit) / params.sigma_logit
        self.assertLess(rel_err_sigma, 0.25,
                        f"σ_logit recovery off: truth={params.sigma_logit}, est={result.sigma_logit}")

    def test_recovers_kappa(self):
        params = SyntheticParams(
            duration_seconds=3600.0, sigma_logit=0.03,
            A_buy=2.0, A_sell=2.0, kappa_buy=60.0, kappa_sell=60.0,
            seed=7,
        )
        result = self._run(params)
        rel_err_kappa = abs(result.kappa_avg - params.kappa_buy) / params.kappa_buy
        self.assertLess(rel_err_kappa, 0.30,
                        f"κ recovery off: truth={params.kappa_buy}, est={result.kappa_avg}")

    def test_recovers_A(self):
        params = SyntheticParams(
            duration_seconds=3600.0, sigma_logit=0.03,
            A_buy=1.5, A_sell=1.5, kappa_buy=50.0, kappa_sell=50.0,
            seed=11,
        )
        result = self._run(params)
        rel_err_A = abs(result.A_avg - params.A_buy) / params.A_buy
        self.assertLess(rel_err_A, 0.50,
                        f"A recovery off: truth={params.A_buy}, est={result.A_avg}")

    def test_asymmetric_arrival_rates(self):
        """If buy-side arrives 3x faster than sell-side, A_buy/A_sell ratio should reflect that.

        We test direction (asymmetry detected) more than magnitude — the
        smaller-A side has fewer trades and exp() of the intercept stderr
        can give wide ratios. A 7200s sim helps tighten this.
        """
        params = SyntheticParams(
            duration_seconds=7200.0, sigma_logit=0.03,
            A_buy=3.0, A_sell=1.0, kappa_buy=50.0, kappa_sell=50.0,
            seed=13,
        )
        result = self._run(params)
        ratio = result.A_buy / result.A_sell
        self.assertGreater(ratio, 1.5, f"asymmetry not detected: ratio={ratio}")
        self.assertLess(ratio, 8.0, f"asymmetry overstated: ratio={ratio}")

    def test_warmup_required(self):
        """Calibrator should refuse to fit on a too-short stream."""
        state = OrderBookState(window_seconds=300.0)
        # Inject just 2 snapshots, no trades.
        state.ingest_snapshot(BookSnapshot(ts=0.0, best_bid=0.49, best_ask=0.51, bid_qty=100, ask_qty=100))
        state.ingest_snapshot(BookSnapshot(ts=1.0, best_bid=0.49, best_ask=0.51, bid_qty=100, ask_qty=100))
        calib = ASCalibrator()
        self.assertIsNone(calib.fit(state))


class TestStateEviction(unittest.TestCase):
    def test_old_data_is_evicted(self):
        state = OrderBookState(window_seconds=10.0)
        for t in range(0, 20):
            state.ingest_snapshot(BookSnapshot(ts=float(t), best_bid=0.49, best_ask=0.51, bid_qty=100, ask_qty=100))
        # Snapshots older than ts=19-10=9 should be gone
        oldest = next(iter(state.snapshots))
        self.assertGreaterEqual(oldest.ts, 9.0)

    def test_crossed_book_skipped(self):
        state = OrderBookState()
        state.ingest_snapshot(BookSnapshot(ts=0.0, best_bid=0.55, best_ask=0.50, bid_qty=100, ask_qty=100))
        self.assertEqual(len(state.snapshots), 0)

    def test_non_open_state_skipped(self):
        state = OrderBookState()
        state.ingest_snapshot(BookSnapshot(ts=0.0, best_bid=0.49, best_ask=0.51, bid_qty=100, ask_qty=100,
                                           state="MARKET_STATE_HALTED"))
        self.assertEqual(len(state.snapshots), 0)


if __name__ == "__main__":
    unittest.main()
