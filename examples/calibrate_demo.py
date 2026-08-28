"""End-to-end demo on a synthetic stream — no API key required.

Runs the simulator → ingests into OrderBookState → fits ASCalibrator → produces
quotes via ASQuoter. Compares fitted (σ, A, κ) to ground truth so you can sanity-
check the bridge without hitting the real Polymarket API.

To run live, swap the simulator section for:

    from polymarket_as.api import APICredentials, PolymarketWSSClient
    creds = APICredentials(access_key=..., secret=...)
    client = PolymarketWSSClient(creds)
    async for msg in client.stream(["your-market-slug"]):
        if isinstance(msg, BookSnapshot):
            state.ingest_snapshot(msg)
        else:
            state.ingest_trade(msg)
        if state.is_warmed_up():
            params = calib.fit(state)
            ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polymarket_as._simulator import SyntheticParams, simulate
from polymarket_as.calibrator import ASCalibrator
from polymarket_as.quoter import ASQuoter
from polymarket_as.state import BookSnapshot, OrderBookState


def run_demo():
    print("=" * 70)
    print("Polymarket × Avellaneda-Stoikov: synthetic-data demo")
    print("=" * 70)

    truth = SyntheticParams(
        duration_seconds=1800.0,   # 30 min
        sigma_logit=0.04,
        A_buy=2.0, kappa_buy=55.0,
        A_sell=1.7, kappa_sell=60.0,
        initial_S=0.55,
        seed=2026,
    )
    print(f"\nGround truth:")
    print(f"  σ_logit       = {truth.sigma_logit}")
    print(f"  A_buy / κ_buy = {truth.A_buy} / {truth.kappa_buy}")
    print(f"  A_sell/ κ_sell= {truth.A_sell} / {truth.kappa_sell}")

    events = simulate(truth)
    n_snaps = sum(1 for e in events if isinstance(e, BookSnapshot))
    n_trades = len(events) - n_snaps
    print(f"\nSimulated stream: {n_snaps} snapshots, {n_trades} trades over {truth.duration_seconds:.0f}s")

    # Ingest into rolling state
    state = OrderBookState(window_seconds=truth.duration_seconds + 10.0)
    for e in events:
        if isinstance(e, BookSnapshot):
            state.ingest_snapshot(e)
        else:
            state.ingest_trade(e)

    # Fit
    calib = ASCalibrator()
    result = calib.fit(state)
    if result is None or not result.is_valid():
        print("\n[FAIL] Calibrator could not fit the synthetic stream.")
        return

    def pct_err(est, truth_v):
        return 100.0 * (est - truth_v) / truth_v

    print(f"\nFitted parameters:")
    print(f"  σ_logit  = {result.sigma_logit:.5f}  (truth {truth.sigma_logit:.5f}, err {pct_err(result.sigma_logit, truth.sigma_logit):+.1f}%)")
    print(f"  A_buy    = {result.A_buy:.3f}     (truth {truth.A_buy:.3f}, err {pct_err(result.A_buy, truth.A_buy):+.1f}%)")
    print(f"  κ_buy    = {result.kappa_buy:.2f}    (truth {truth.kappa_buy:.2f}, err {pct_err(result.kappa_buy, truth.kappa_buy):+.1f}%)")
    print(f"  A_sell   = {result.A_sell:.3f}     (truth {truth.A_sell:.3f}, err {pct_err(result.A_sell, truth.A_sell):+.1f}%)")
    print(f"  κ_sell   = {result.kappa_sell:.2f}    (truth {truth.kappa_sell:.2f}, err {pct_err(result.kappa_sell, truth.kappa_sell):+.1f}%)")
    print(f"  σ_price@S= {result.sigma_price:.5f}  (chain rule σ_logit · S(1-S))")
    print(f"  fit window: {result.fit_horizon_seconds:.0f}s, {result.n_snapshots} snapshots")
    print(f"             trades: BUY={result.n_trades_buy}, SELL={result.n_trades_sell}")

    # Demonstrate quote generation under various inventory & horizon scenarios
    quoter = ASQuoter(gamma=0.1, tick_size=0.01)
    latest = state.latest()
    S = latest.micro if latest and latest.micro else 0.5
    print(f"\nCurrent micro-price: S = {S:.4f}")
    print(f"\nQuotes under different states:")
    print(f"  {'inventory':>10}  {'horizon':>10}  {'mode':>14}  {'bid':>6}  {'ask':>6}  {'res':>6}  {'half-spread':>12}")
    scenarios = [
        ("flat, 1h",      0.0,  3600.0),
        ("long 50, 1h",   50.0, 3600.0),
        ("short 50, 1h", -50.0, 3600.0),
        ("long 50, 1d",   50.0, 86_400.0),
        ("long 50, 5min", 50.0, 300.0),     # close-out mode
    ]
    for label, inv, horizon in scenarios:
        q = quoter.quote(result, S=S, inventory=inv, time_to_resolution_seconds=horizon)
        bid = f"{q.bid:.3f}" if q.bid is not None else "—"
        ask = f"{q.ask:.3f}" if q.ask is not None else "—"
        print(f"  {label:>10}  {horizon:>9.0f}s  {q.mode:>14}  {bid:>6}  {ask:>6}  {q.reservation:>6.3f}  {q.half_spread:>12.4f}")


if __name__ == "__main__":
    run_demo()
