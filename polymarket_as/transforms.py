"""Numerical transforms used throughout the calibrator."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Boundary clamp: logit(0) and logit(1) blow up. Anything inside [eps, 1-eps] is fine.
_LOGIT_EPS = 1e-6


def logit(p):
    """Logit transform with safe clipping."""
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def expit(x):
    """Inverse logit (sigmoid)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def microprice(
    best_bid: float,
    best_ask: float,
    bid_qty: float,
    ask_qty: float,
) -> Optional[float]:
    """Queue-imbalance weighted mid.

    Returns None if both sides have zero (or negative) depth — caller should
    fall back to last-trade or skip the snapshot.
    """
    if bid_qty <= 0 and ask_qty <= 0:
        return None
    if bid_qty <= 0:
        return best_ask
    if ask_qty <= 0:
        return best_bid
    return (ask_qty * best_bid + bid_qty * best_ask) / (bid_qty + ask_qty)


def round_to_tick(price: float, tick: float) -> float:
    """Round to the nearest tick. Polymarket: tick = $0.01 (or $0.001 on some markets)."""
    return round(price / tick) * tick


def clip_price(price: float, tick: float = 0.01) -> float:
    """Clip price to [tick, 1 - tick] — Polymarket markets reject quotes at exactly 0 or 1."""
    return min(1.0 - tick, max(tick, price))


def price_space_sigma(sigma_logit: float, s: float) -> float:
    """Convert logit-space σ to price-space σ at level S using chain rule.

    dS = S(1-S) dx, so for x ~ N(0, σ²) locally, S ~ N(s, (s(1-s))² σ²).
    """
    s = float(s)
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return sigma_logit * s * (1.0 - s)
