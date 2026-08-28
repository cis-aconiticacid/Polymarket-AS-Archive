"""Estimate (σ, A, κ) from rolling window of book snapshots + trades.

See ENG_NOTE.md §4–5 for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np

from polymarket_as.state import OrderBookState
from polymarket_as.transforms import logit, price_space_sigma


@dataclass(frozen=True)
class CalibrationResult:
    sigma_logit: float            # σ in logit-space, per √second
    sigma_price: float            # σ_S(S_t) at the current S, per √second
    A_buy: float                  # ask-side intensity scale (taker BUY hits ask)
    kappa_buy: float
    A_sell: float                 # bid-side intensity scale (taker SELL hits bid)
    kappa_sell: float
    A_avg: float                  # symmetric average — what AS quoter uses
    kappa_avg: float
    n_snapshots: int
    n_trades_buy: int
    n_trades_sell: int
    fit_horizon_seconds: float

    def is_valid(self) -> bool:
        return (
            self.sigma_logit > 0.0
            and self.kappa_avg > 0.0
            and self.A_avg > 0.0
            and math.isfinite(self.sigma_logit)
            and math.isfinite(self.kappa_avg)
            and math.isfinite(self.A_avg)
        )


@dataclass
class ASCalibrator:
    """Fits AS-flavored parameters from an `OrderBookState`."""

    resample_seconds: float = 5.0       # σ̂ resampling grid
    delta_bin_width: float = 0.005      # 0.5¢ bins for κ̂ histogram
    max_delta: float = 0.20             # only use trades within ±20¢ of mid
    min_count_per_bin: int = 3          # κ̂ regression skips sparser bins
    min_bins_for_fit: int = 3           # need at least this many bins for OLS

    def fit(self, state: OrderBookState, current_S: Optional[float] = None) -> Optional[CalibrationResult]:
        if not state.is_warmed_up():
            return None
        snaps = list(state.iter_snapshots())
        trades = list(state.iter_trades())
        if len(snaps) < 2:
            return None

        sigma_logit = self._estimate_sigma_logit(snaps)
        if not (sigma_logit > 0 and math.isfinite(sigma_logit)):
            return None

        if current_S is None:
            latest = state.latest()
            current_S = latest.micro if latest and latest.micro is not None else (latest.mid if latest else 0.5)

        sigma_price = price_space_sigma(sigma_logit, current_S)

        A_buy, kappa_buy, n_buy = self._estimate_lambda(state, trades, side="BUY")
        A_sell, kappa_sell, n_sell = self._estimate_lambda(state, trades, side="SELL")

        # Symmetric average for the AS formula — but only over the sides that
        # actually fit. Falls back to whichever side is valid; if neither, bail.
        valids: List[Tuple[float, float]] = []
        if A_buy is not None and kappa_buy is not None:
            valids.append((A_buy, kappa_buy))
        if A_sell is not None and kappa_sell is not None:
            valids.append((A_sell, kappa_sell))
        if not valids:
            return None
        A_avg = float(np.mean([a for a, _ in valids]))
        kappa_avg = float(np.mean([k for _, k in valids]))

        horizon = snaps[-1].ts - snaps[0].ts
        return CalibrationResult(
            sigma_logit=sigma_logit,
            sigma_price=sigma_price,
            A_buy=A_buy if A_buy is not None else float("nan"),
            kappa_buy=kappa_buy if kappa_buy is not None else float("nan"),
            A_sell=A_sell if A_sell is not None else float("nan"),
            kappa_sell=kappa_sell if kappa_sell is not None else float("nan"),
            A_avg=A_avg,
            kappa_avg=kappa_avg,
            n_snapshots=len(snaps),
            n_trades_buy=n_buy,
            n_trades_sell=n_sell,
            fit_horizon_seconds=horizon,
        )

    # -- σ̂ -----------------------------------------------------------------
    def _estimate_sigma_logit(self, snaps) -> float:
        """Realized variance of logit-microprice on a regular grid.

        Uses the simple mean estimator (Δx² averaged then scaled by Δt), with
        a winsorize at the 99.5th percentile to suppress one-off jumps. The
        often-cited median-RV is biased to zero under tick-discretization
        (most consecutive 5s bars on Polymarket have identical microprices),
        so we don't use it here. For jump-heavy markets, swap in bipower
        variation if needed.
        """
        ts = np.array([s.ts for s in snaps])
        mids = np.array([s.micro if s.micro is not None else s.mid for s in snaps])
        t0, t1 = ts[0], ts[-1]
        if t1 <= t0:
            return 0.0
        grid = np.arange(t0, t1, self.resample_seconds)
        if grid.size < 4:
            return 0.0
        idx = np.searchsorted(ts, grid, side="right") - 1
        idx = np.clip(idx, 0, len(ts) - 1)
        x = logit(mids[idx])
        dx = np.diff(x)
        if dx.size < 4:
            return 0.0
        sq = dx ** 2
        # Winsorize top 0.5% (jump suppression). For < 200 obs this is a no-op.
        if sq.size >= 200:
            cap = np.quantile(sq, 0.995)
            sq = np.minimum(sq, cap)
        var_per_sec = float(np.mean(sq)) / self.resample_seconds
        return float(math.sqrt(max(var_per_sec, 0.0)))

    # -- (Â, κ̂) -----------------------------------------------------------
    def _estimate_lambda(self, state: OrderBookState, trades, side: str):
        """Fit ln λ̂(δ) = ln A − κ δ for one side. Returns (A, κ, n_trades_on_side)."""
        deltas = []
        for tr in trades:
            if tr.taker_side != side:
                continue
            snap = state.snapshot_at(tr.ts)
            if snap is None:
                continue
            mid = snap.micro if snap.micro is not None else snap.mid
            d = abs(tr.price - mid)
            if d >= self.max_delta:
                continue
            deltas.append(d)
        n = len(deltas)
        if n < self.min_bins_for_fit * self.min_count_per_bin:
            return None, None, n
        deltas = np.array(deltas)

        # Bin edges: 0, w, 2w, ..., max_delta.
        nbins = int(math.ceil(self.max_delta / self.delta_bin_width))
        edges = np.linspace(0.0, nbins * self.delta_bin_width, nbins + 1)
        counts, _ = np.histogram(deltas, bins=edges)

        # Fit horizon — span of trades, not the whole state buffer.
        # Empty side could give horizon=0; default to state window then.
        if state.trades:
            horizon = state.trades[-1].ts - state.trades[0].ts
        else:
            horizon = state.window_seconds
        if horizon <= 0.0:
            horizon = state.window_seconds

        # Density estimate: λ(δ) is the per-second per-unit-δ intensity.
        # Histogram count over a bin of width w corresponds to T·w·λ̄, so
        # density λ̂(δ) = count / (T · w). Without the /w we'd recover w·A
        # instead of A.
        density = counts / (horizon * self.delta_bin_width)
        centers = 0.5 * (edges[:-1] + edges[1:])

        keep = counts >= self.min_count_per_bin
        if keep.sum() < self.min_bins_for_fit:
            return None, None, n
        log_rates = np.log(density[keep])
        x = centers[keep]
        # OLS: log_rates = ln A − κ x  ⇒  slope = −κ, intercept = ln A
        slope, intercept = np.polyfit(x, log_rates, 1)
        kappa = float(-slope)
        A = float(math.exp(intercept))
        if kappa <= 0.0 or not math.isfinite(kappa):
            return None, None, n
        if A <= 0.0 or not math.isfinite(A):
            return None, None, n
        return A, kappa, n
