"""Avellaneda-Stoikov quoter, adapted for Polymarket boundaries and ticks.

Closed-form formulas from Avellaneda & Stoikov (2008), with engineering
adjustments:
  - σ in price space evaluated at current S (frozen-coefficient hack)
  - (T − t) clamped to [0, MAX_HORIZON]
  - Close-out mode near resolution
  - One-sided quoting near {0, 1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from polymarket_as.calibrator import CalibrationResult
from polymarket_as.transforms import clip_price, round_to_tick


@dataclass(frozen=True)
class Quote:
    bid: Optional[float]
    ask: Optional[float]
    reservation: float
    half_spread: float
    mode: str  # 'normal' | 'close_out' | 'one_sided_low' | 'one_sided_high' | 'invalid'


@dataclass
class ASQuoter:
    gamma: float = 0.1                          # risk aversion (user choice)
    tick_size: float = 0.01
    max_horizon_seconds: float = 86_400.0       # cap (T − t)
    close_out_threshold_seconds: float = 600.0  # below this, switch to close-out
    boundary_low: float = 0.05                  # one-sided below this
    boundary_high: float = 0.95                 # one-sided above this

    def quote(
        self,
        params: CalibrationResult,
        S: float,
        inventory: float,
        time_to_resolution_seconds: float,
    ) -> Quote:
        if not params.is_valid():
            return Quote(None, None, S, 0.0, "invalid")

        # Effective horizon used by the inventory penalty.
        T_t = max(0.0, min(time_to_resolution_seconds, self.max_horizon_seconds))

        if time_to_resolution_seconds <= self.close_out_threshold_seconds:
            return self._close_out_quote(S, inventory, params)

        sigma2 = params.sigma_price ** 2
        gamma = self.gamma
        kappa = params.kappa_avg

        # Reservation price: r = S − q γ σ² (T − t)
        reservation = S - inventory * gamma * sigma2 * T_t
        # Optimal half-spread: δ* = (γ σ² (T − t)) / 2 + (1/γ) ln(1 + γ/κ)
        delta = 0.5 * gamma * sigma2 * T_t + math.log1p(gamma / kappa) / gamma

        bid_raw = reservation - delta
        ask_raw = reservation + delta

        # Boundary handling: if the reservation is near {0, 1}, AS spreads are
        # negative-EV on the side facing the boundary. Quote one-sided.
        if reservation < self.boundary_low:
            # Low fair value — only bid (very tight); skip ask.
            bid = clip_price(round_to_tick(bid_raw, self.tick_size), self.tick_size)
            return Quote(bid=bid, ask=None, reservation=reservation, half_spread=delta,
                         mode="one_sided_low")
        if reservation > self.boundary_high:
            ask = clip_price(round_to_tick(ask_raw, self.tick_size), self.tick_size)
            return Quote(bid=None, ask=ask, reservation=reservation, half_spread=delta,
                         mode="one_sided_high")

        bid = clip_price(round_to_tick(bid_raw, self.tick_size), self.tick_size)
        ask = clip_price(round_to_tick(ask_raw, self.tick_size), self.tick_size)
        # Ensure bid < ask after rounding; if collapsed, widen by one tick.
        if ask <= bid:
            ask = clip_price(bid + self.tick_size, self.tick_size)
        return Quote(bid=bid, ask=ask, reservation=reservation, half_spread=delta,
                     mode="normal")

    def _close_out_quote(self, S: float, inventory: float, params: CalibrationResult) -> Quote:
        """Aggressive inventory unwind: tighten the side that reduces exposure."""
        # One-tick base spread, skewed toward unwinding direction.
        base = self.tick_size
        if inventory > 0:
            # Long → want to sell. Tight ask, wide bid.
            ask = clip_price(round_to_tick(S, self.tick_size), self.tick_size)
            bid = clip_price(round_to_tick(S - 5 * base, self.tick_size), self.tick_size)
        elif inventory < 0:
            ask = clip_price(round_to_tick(S + 5 * base, self.tick_size), self.tick_size)
            bid = clip_price(round_to_tick(S, self.tick_size), self.tick_size)
        else:
            ask = clip_price(round_to_tick(S + base, self.tick_size), self.tick_size)
            bid = clip_price(round_to_tick(S - base, self.tick_size), self.tick_size)
        if ask <= bid:
            ask = clip_price(bid + self.tick_size, self.tick_size)
        return Quote(bid=bid, ask=ask, reservation=S, half_spread=base, mode="close_out")
