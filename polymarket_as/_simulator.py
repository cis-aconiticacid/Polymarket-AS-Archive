"""Synthetic stream generator for tests and offline demos.

Underscore prefix: this is internal — for validation, not a production simulator.
Generates a stream of BookSnapshot + TradeEvent objects with known
ground-truth (σ_logit, A, κ) so the calibrator can be checked against truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Union

import numpy as np

from polymarket_as.state import BookSnapshot, TradeEvent
from polymarket_as.transforms import expit, logit


@dataclass
class SyntheticParams:
    duration_seconds: float = 600.0    # 10 min default for tests
    snapshot_interval: float = 1.0     # 1 Hz book updates (debounced WSS feel)
    sigma_logit: float = 0.05          # √(per second)
    A_buy: float = 1.5                 # arrivals/sec at δ=0
    kappa_buy: float = 60.0            # decay per unit price (so 60 ≈ steep at $0.05)
    A_sell: float = 1.5
    kappa_sell: float = 60.0
    initial_S: float = 0.5
    spread: float = 0.02               # constant best-ask − best-bid (synthetic)
    side_qty: float = 100.0            # constant L1 quantity each side
    tick: float = 0.01
    max_trade_distance: float = 0.10   # cap how far from mid trades can occur
    seed: int = 0


def simulate(params: SyntheticParams) -> List[Union[BookSnapshot, TradeEvent]]:
    """Generate a time-ordered list of book snapshots and trades."""
    rng = np.random.default_rng(params.seed)

    n_steps = int(params.duration_seconds / params.snapshot_interval)
    # Logit-BM with known σ
    increments = rng.normal(0.0, params.sigma_logit * np.sqrt(params.snapshot_interval), n_steps)
    x = logit(np.array([params.initial_S]))[0] + np.cumsum(np.concatenate([[0.0], increments]))
    S = expit(x)  # length n_steps + 1

    events: List[Union[BookSnapshot, TradeEvent]] = []

    # Build snapshot timeline
    half = params.spread / 2.0
    for k in range(n_steps + 1):
        t = k * params.snapshot_interval
        s = float(S[k])
        bb = max(params.tick, _round(s - half, params.tick))
        ba = min(1.0 - params.tick, _round(s + half, params.tick))
        if ba <= bb:
            ba = min(1.0 - params.tick, bb + params.tick)
        events.append(BookSnapshot(
            ts=t,
            best_bid=bb, best_ask=ba,
            bid_qty=params.side_qty, ask_qty=params.side_qty,
        ))

    # Generate trades by thinning a Poisson process per side.
    # Per side: λ_total = ∫_0^max_distance A exp(-κ δ) dδ = A (1 - exp(-κ M)) / κ
    for side, A, kappa in [("BUY", params.A_buy, params.kappa_buy),
                           ("SELL", params.A_sell, params.kappa_sell)]:
        lam_total = A * (1.0 - np.exp(-kappa * params.max_trade_distance)) / kappa
        n_expected = lam_total * params.duration_seconds
        n_trades = rng.poisson(n_expected)
        ts_arr = np.sort(rng.uniform(0.0, params.duration_seconds, n_trades))
        # Sample δ from truncated exponential with rate κ on [0, max_distance].
        u = rng.uniform(0.0, 1.0, n_trades)
        delta_arr = -np.log(1.0 - u * (1.0 - np.exp(-kappa * params.max_trade_distance))) / kappa
        for t, d in zip(ts_arr, delta_arr):
            # Find the snapshot at-or-before t
            idx = min(int(t / params.snapshot_interval), n_steps)
            mid = float(S[idx])
            if side == "BUY":
                price = min(1.0 - params.tick, _round(mid + d, params.tick))
            else:
                price = max(params.tick, _round(mid - d, params.tick))
            events.append(TradeEvent(ts=float(t), price=price, qty=10.0, taker_side=side))

    events.sort(key=lambda e: e.ts)
    return events


def _round(x: float, tick: float) -> float:
    return round(x / tick) * tick


def split_into_state_inputs(events: List[Union[BookSnapshot, TradeEvent]]) -> Tuple[List[BookSnapshot], List[TradeEvent]]:
    """Convenience: split mixed-event list into homogeneous lists."""
    snaps = [e for e in events if isinstance(e, BookSnapshot)]
    trades = [e for e in events if isinstance(e, TradeEvent)]
    return snaps, trades
