"""Tests for the AS quoter behavior."""

import math
import unittest

from polymarket_as.calibrator import CalibrationResult
from polymarket_as.quoter import ASQuoter


def make_params(sigma_price=0.001, A=1.0, kappa=50.0):
    """Realistic-scale fixture: σ_price = 0.001/√s ≈ 6%/√h, typical of moderately liquid markets."""
    return CalibrationResult(
        sigma_logit=sigma_price * 4.0,  # arbitrary; quoter uses sigma_price
        sigma_price=sigma_price,
        A_buy=A, kappa_buy=kappa,
        A_sell=A, kappa_sell=kappa,
        A_avg=A, kappa_avg=kappa,
        n_snapshots=100, n_trades_buy=50, n_trades_sell=50,
        fit_horizon_seconds=300.0,
    )


class TestQuoter(unittest.TestCase):
    def test_zero_inventory_is_symmetric(self):
        q = ASQuoter(gamma=0.1, tick_size=0.01).quote(
            make_params(), S=0.5, inventory=0.0, time_to_resolution_seconds=3600.0,
        )
        self.assertEqual(q.mode, "normal")
        spread_above = q.ask - 0.5
        spread_below = 0.5 - q.bid
        self.assertAlmostEqual(spread_above, spread_below, places=2)

    def test_long_inventory_skews_lower(self):
        """If we're long, reservation price drops below S, so quotes shift down.

        Realistic Polymarket scale: σ_price ≈ 0.0005/√s (~3¢ std over an hour).
        γ=5 and q=20 produce a ~1¢ shift which is detectable after tick-rounding.
        """
        params = make_params(sigma_price=0.0005)
        q_neutral = ASQuoter(gamma=5.0, tick_size=0.01).quote(
            params, S=0.5, inventory=0.0, time_to_resolution_seconds=3600.0,
        )
        q_long = ASQuoter(gamma=5.0, tick_size=0.01).quote(
            params, S=0.5, inventory=20.0, time_to_resolution_seconds=3600.0,
        )
        self.assertLess(q_long.reservation, q_neutral.reservation)
        self.assertLessEqual(q_long.bid, q_neutral.bid)
        self.assertLessEqual(q_long.ask, q_neutral.ask)
        # And at least one side must move (otherwise no skew).
        self.assertTrue(q_long.bid < q_neutral.bid or q_long.ask < q_neutral.ask)

    def test_close_out_mode_when_near_resolution(self):
        q = ASQuoter().quote(
            make_params(), S=0.5, inventory=5.0, time_to_resolution_seconds=60.0,
        )
        self.assertEqual(q.mode, "close_out")
        # When long, ask should be tight (= S rounded)
        self.assertAlmostEqual(q.ask, 0.5, places=2)
        # And bid should be much further away
        self.assertLess(q.bid, q.ask - 0.04)

    def test_one_sided_at_low_boundary(self):
        # Reservation around 0.03 → only bid should be quoted
        q = ASQuoter().quote(
            make_params(), S=0.03, inventory=0.0, time_to_resolution_seconds=3600.0,
        )
        self.assertEqual(q.mode, "one_sided_low")
        self.assertIsNone(q.ask)
        self.assertIsNotNone(q.bid)

    def test_one_sided_at_high_boundary(self):
        q = ASQuoter().quote(
            make_params(), S=0.97, inventory=0.0, time_to_resolution_seconds=3600.0,
        )
        self.assertEqual(q.mode, "one_sided_high")
        self.assertIsNone(q.bid)
        self.assertIsNotNone(q.ask)

    def test_horizon_clamped(self):
        """Setting absurd horizon (10 days) should not blow up spread vs 1-day horizon."""
        q_one_day = ASQuoter().quote(
            make_params(), S=0.5, inventory=10.0, time_to_resolution_seconds=86_400.0,
        )
        q_ten_days = ASQuoter().quote(
            make_params(), S=0.5, inventory=10.0, time_to_resolution_seconds=10 * 86_400.0,
        )
        self.assertAlmostEqual(q_one_day.reservation, q_ten_days.reservation, places=8)
        self.assertAlmostEqual(q_one_day.half_spread, q_ten_days.half_spread, places=8)

    def test_invalid_calibration_returns_no_quotes(self):
        bad = CalibrationResult(
            sigma_logit=0.0, sigma_price=0.0,
            A_buy=float("nan"), kappa_buy=float("nan"),
            A_sell=float("nan"), kappa_sell=float("nan"),
            A_avg=float("nan"), kappa_avg=float("nan"),
            n_snapshots=0, n_trades_buy=0, n_trades_sell=0,
            fit_horizon_seconds=0.0,
        )
        q = ASQuoter().quote(bad, S=0.5, inventory=0.0, time_to_resolution_seconds=3600.0)
        self.assertEqual(q.mode, "invalid")
        self.assertIsNone(q.bid)
        self.assertIsNone(q.ask)

    def test_bid_strictly_below_ask(self):
        # Even at tiny half-spreads, bid < ask after rounding.
        q = ASQuoter(tick_size=0.01).quote(
            make_params(sigma_price=0.001), S=0.5, inventory=0.0, time_to_resolution_seconds=3600.0,
        )
        self.assertLess(q.bid, q.ask)


if __name__ == "__main__":
    unittest.main()
