# polymarket_as

Engineering bridge from Polymarket CLOB API streams to Avellaneda-Stoikov market-making parameters.

**Not a research / publication project — a working library that respects what Polymarket's data shape actually is and what AS's formulas actually need.**

## Quickstart

```bash
# Run the end-to-end synthetic demo (no API key needed)
python3 examples/calibrate_demo.py

# Run the test suite
python3 -m unittest discover -s tests
```

Expected demo output: σ_logit recovered within ~3% of ground truth, κ within ~25%, A within ~55% over a 30-min sim. Quotes shown under flat, long, short, and close-out scenarios.

## What this gives you

```python
from polymarket_as import OrderBookState, ASCalibrator, ASQuoter, BookSnapshot, TradeEvent

state = OrderBookState(window_seconds=300)        # rolling 5-min buffer
calib = ASCalibrator()                            # fits σ̂, Â, κ̂
quoter = ASQuoter(gamma=0.1, tick_size=0.01)      # produces (bid, ask)

# In your WSS loop:
state.ingest_snapshot(snap)   # or state.ingest_trade(trade)
if state.is_warmed_up():
    params = calib.fit(state)
    if params is not None and params.is_valid():
        q = quoter.quote(params, S=current_S, inventory=q,
                         time_to_resolution_seconds=T_minus_t)
        # q.bid, q.ask — None on close-out / boundary one-sided cases
```

## Notes

Polymarket gives you bounded prices in `(0, 1)`, depth-aggregated book snapshots (no L3), maker/taker-tagged trades, and resolution events that drag the price to `{0, 1}`. AS assumes unbounded arithmetic Brownian motion, constant volatility, and Poisson order arrivals at any distance from mid. You can't just plug `(bestBid+bestAsk)/2` and rolling-RV into the AS formulas.

See **`ENG_NOTE.md`** for the full parameter-by-parameter mapping, the logit-transform argument, the trade-distance histogram fit for `(A, κ)`, and the boundary / close-out / horizon-clamp engineering decisions.

## Files

| Path | Purpose |
|---|---|
| `ENG_NOTE.md` | Design rationale, math, approximations, validation plan |
| `polymarket_as/transforms.py` | `logit`/`expit`/`microprice`/`round_to_tick`/`clip_price` |
| `polymarket_as/state.py` | `OrderBookState`, `BookSnapshot`, `TradeEvent` |
| `polymarket_as/calibrator.py` | `ASCalibrator` → `(σ_logit, σ_price, A, κ)` per side |
| `polymarket_as/quoter.py` | `ASQuoter` → AS bid/ask with boundary + close-out logic |
| `polymarket_as/api.py` | REST + WSS client (HMAC signing); requires `requests`, `websockets` |
| `polymarket_as/_simulator.py` | Synthetic generator with known ground-truth — for tests/demo only |
| `examples/calibrate_demo.py` | End-to-end synthetic demo |
| `tests/` | unittest suite (no pytest dependency) |

## Limitations (read these)

- Constant-σ AS used with state-dependent `σ_S(S) = σ_logit · S(1−S)` — a frozen-coefficient hack, not a re-derivation. Valid when S doesn't wander far over a quote's lifetime.
- `(A, κ)` fit is on **taker arrival distance histogram**, an upper bound on fill rate. Queue-position correction left for execution-time.
- `γ` is not estimable — user-supplied.
- No jump model. News-event spikes are partially handled by winsorizing σ̂; better strategies should explicitly pause around scheduled events.
- WSS subscriber requires `pip install websockets`. Synthetic demo + tests don't need it.
- This doesn't give a rigorous bounded-diffusion treatment (Jacobi process etc.) that re-derives AS from scratch for prediction markets. That's a research project, not engineering.
