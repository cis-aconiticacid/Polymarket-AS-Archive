"""Stateful aggregator that ingests Polymarket WSS messages and feeds the calibrator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional

from polymarket_as.transforms import microprice


@dataclass(frozen=True)
class BookSnapshot:
    """A single book snapshot. Times are in seconds (epoch or simulated)."""

    ts: float
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    state: str = "MARKET_STATE_OPEN"

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def micro(self) -> Optional[float]:
        return microprice(self.best_bid, self.best_ask, self.bid_qty, self.ask_qty)


@dataclass(frozen=True)
class TradeEvent:
    """A trade event. `taker_side` is one of 'BUY' or 'SELL'.

    `taker_side='BUY'` means the taker bought (lifted the offer) → ask-side fill.
    `taker_side='SELL'` means the taker sold (hit the bid) → bid-side fill.
    """

    ts: float
    price: float
    qty: float
    taker_side: str  # 'BUY' or 'SELL'


@dataclass
class OrderBookState:
    """Rolling-window store of book snapshots and trades.

    Window length controls how much history the calibrator sees on each fit.
    Default 300s = 5 minutes is a reasonable starting point for moderately
    active markets; tune up for thin markets, down for very active ones.
    """

    window_seconds: float = 300.0
    snapshots: Deque[BookSnapshot] = field(default_factory=deque)
    trades: Deque[TradeEvent] = field(default_factory=deque)
    _last_ts: float = field(default=-float("inf"))

    def ingest_snapshot(self, snap: BookSnapshot) -> None:
        if snap.state != "MARKET_STATE_OPEN":
            # Suspended/halted/expired markets break stationarity.
            return
        if snap.best_ask <= snap.best_bid:
            # Crossed/locked book — skip rather than ingest garbage.
            return
        self.snapshots.append(snap)
        self._last_ts = max(self._last_ts, snap.ts)
        self._evict()

    def ingest_trade(self, trade: TradeEvent) -> None:
        if trade.taker_side not in ("BUY", "SELL"):
            return
        self.trades.append(trade)
        self._last_ts = max(self._last_ts, trade.ts)
        self._evict()

    def _evict(self) -> None:
        cutoff = self._last_ts - self.window_seconds
        while self.snapshots and self.snapshots[0].ts < cutoff:
            self.snapshots.popleft()
        while self.trades and self.trades[0].ts < cutoff:
            self.trades.popleft()

    def is_warmed_up(self, min_snaps: int = 30, min_trades: int = 20) -> bool:
        return len(self.snapshots) >= min_snaps and len(self.trades) >= min_trades

    def snapshot_at(self, ts: float) -> Optional[BookSnapshot]:
        """Return the most recent snapshot at or before `ts`. None if no history."""
        if not self.snapshots:
            return None
        # Linear scan from the right; trades typically come in close to live.
        for s in reversed(self.snapshots):
            if s.ts <= ts:
                return s
        return None

    def latest(self) -> Optional[BookSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    def iter_snapshots(self) -> Iterable[BookSnapshot]:
        return iter(self.snapshots)

    def iter_trades(self) -> Iterable[TradeEvent]:
        return iter(self.trades)
