# Polymarket → Avellaneda-Stoikov: Engineering Bridge

**Status:** engineering, not publication. Goal: a working calibrator that turns Polymarket CLOB API streams into the parameter set Avellaneda-Stoikov (AS) needs, with documented approximations.

## 1. The mismatch

AS (Avellaneda & Stoikov 2008) assumes:

- Mid-price `S_t` is arithmetic Brownian motion: `dS = σ dW`, σ constant.
- Order arrivals at distance `δ` from mid are Poisson with intensity `λ(δ) = A · exp(-κ · δ)`.
- Closed-form quotes:
  - reservation price `r(s, q, t) = s − q · γ · σ² · (T − t)`
  - half-spread `δ* = (γ σ² (T − t))/2 + (1/γ) ln(1 + γ/κ)`

Polymarket gives us, instead:

- A bounded price `S_t ∈ (0, 1)` that converges to `{0, 1}` at resolution time `T_resolve`.
- A depth-aggregated CLOB (book snapshots, no L3 quote stream).
- A trade tape with **maker/taker side and intent** tags (`BUY_LONG / SELL_LONG / BUY_SHORT / SELL_SHORT`).
- Tick size $0.01 on most markets, $0.001 on some.
- Two REST endpoints (`/book`, `/bbo`) at low rate-limit; one WebSocket with `MARKET_DATA`, `MARKET_DATA_LITE`, `TRADE` channels.

You **cannot** plug `(bestBid+bestAsk)/2` and a rolling realized variance into the AS formulas and expect anything meaningful. The boundedness inflates σ near boundaries, the wide spreads break the small-`δ` regime AS lives in, and constant-σ assumption is silly when variance is structurally going to zero at resolution.

## 2. Parameter-by-parameter mapping

| AS parameter   | Polymarket source                       | Estimator (this repo)                                    | Caveat                                                 |
|---             |---                                      |---                                                       |---                                                     |
| `S_t` (mid)    | `bestBid`, `bestAsk`, depth at L1       | `microprice = (askQty·bid + bidQty·ask)/(bidQty+askQty)` | falls back to mid if both sides empty                  |
| `σ`            | logit-transformed mid-price increments  | median realized variance on `Δx`, jump-robust            | reported as `σ_logit`; converted to price-space at use |
| `A`, `κ`       | trade tape distance-from-mid histogram  | OLS on `ln λ̂(δ) = ln A − κ δ` per side                  | upper-bound on fill rate (FIFO/queue not modeled)      |
| `γ`            | user input                              | constant                                                 | not estimable                                          |
| `T − t`        | `closeSetTime` (resolution timestamp)   | clamped to `[0, max_horizon_seconds]`                    | switches to close-out mode near resolution             |
| tick / clip    | exchange minimum                        | round to tick; clip to `[0.01, 0.99]`; one-sided at edge | enforce YES+NO = 1 self-consistency                    |

## 3. Why the logit transform

Define `x_t = logit(S_t) = ln(S_t / (1 − S_t))`. Then `x_t ∈ ℝ` and the chain rule gives
`dx = (1 / (S(1−S))) dS`. If we *assume* `dx ≈ σ_logit dW` is locally Gaussian (a plausible local model for prediction-market dynamics that resembles a Brier-score-driven update process), then:

- `σ_logit` is a stable quantity to estimate by RV.
- The price-space volatility is `σ_S(S) = σ_logit · S · (1 − S)`. It naturally collapses at the boundaries.
- AS reservation/spread formulas use `σ²` evaluated *at the current S* — we plug `σ_S(S_t)²` and get a state-dependent spread that tightens at boundaries (correct) without us re-deriving anything.

**Trade-off:** the AS derivation assumes constant σ. Using `σ_S(S_t)` is an engineering hack — call it a "frozen-coefficient" approximation, valid when `S` doesn't move much over the order's lifetime. For the inventory-bearing horizon `(T − t)` term the approximation is poorer; we account for that by clamping `(T − t)` to a moderate value (see §6).

## 4. σ̂ estimator

Inputs: a stream of book snapshots `{ts, bid, ask, bidQty, askQty}`.

1. Compute microprice `S_t` per snapshot.
2. Resample to fixed grid (default 5s) — last-observation-carried-forward.
3. Compute `x_t = logit(S_t)` and increments `Δx_t`.
4. Estimate `σ²_logit` using **median realized variance** (Andersen-Dobrev-Schaumburg style):

   `σ²_logit ≈ (π / (6 − 4√3 + π)) · median(Δx_t²) · (1 / Δt)`

   The constant ≈ 1.4826 makes this a consistent estimator of integrated variance under jumps, much more robust than plain RV in news-driven prediction markets.
5. Convert per estimate use: `σ_S(S_t) = σ_logit · S_t · (1 − S_t)`.

## 5. (A, κ) estimator from trades

The AS `λ(δ) = A exp(−κδ)` says: intensity of *fills* of a passive quote at distance `δ` from mid. We don't directly observe fills of our hypothetical quotes, but we can observe **taker arrivals** as a proxy for the upper-envelope intensity at each price level.

Per WSS `TRADE` message we have `{price, quantity, tradeTime, maker.side, taker.side}`. The taker side is the side that hit the book:

- `taker.side == ORDER_SIDE_BUY` → liquidity removed from the **ask**; this is an arrival "at the ask side".
- `taker.side == ORDER_SIDE_SELL` → liquidity removed from the **bid** side.

Define per-trade distance-from-mid as `δ = |price − S_t|` measured against the snapshot mid at trade time. Bin by `δ` (default 0.5¢ bins). Over a window of length `W`, the empirical intensity in bin `[δ, δ+dδ]` per side is:

   `λ̂_side(δ) = N_side(δ_bin) / W`

Fit on log scale (only bins with `N ≥ min_count`):

   `ln λ̂_side(δ) = ln A_side − κ_side · δ + ε`

OLS gives `(Â_side, κ̂_side)`. Drop bins below `min_count` (default 3) to avoid log(0).

**What this estimator does NOT capture:**

- **Queue position.** In a thick book, a taker-arrival event consumes only the front of the queue, not your order at the back. Mitigation: at quote-time, divide effective intensity by `(1 + bookDepthAtLevel / yourQuoteSize)`. Implemented as `effective_lambda(δ) = λ̂(δ) · 1/(1 + q_book/q_us)` in `quoter.py`, but the calibrator returns the raw fit.
- **Same-side cancel-replace flow.** A market order arriving cancels passive orders' opportunity cost; pure cancellations don't. Without an L3 stream we can't separate. Practically: passive cancel rate appears as elevated κ (steep decay = quotes far from mid are stale and don't fill). Live with it.
- **Asymmetry.** `Â_buy ≠ Â_sell` carries directional information. We expose both sides separately; the AS formulas use the average for symmetric quoting, but a directional-aware quoter would use them separately.

**Sanity:** in stationary markets, `Â_buy ≈ Â_sell` and `κ̂_buy ≈ κ̂_sell`. Persistent asymmetry → shift `S` toward the cheaper side, or tighten one side and widen the other.

## 6. (T − t) and close-out mode

Polymarket markets have a known resolution time. For AS, the `(T − t)` term controls the inventory-penalty's growth: longer horizon = more risk per unit inventory = wider spreads + bigger reservation skew.

Pure AS: as `t → T`, the formulas collapse correctly (penalty → 0) and the optimal strategy becomes "quote at fair, infinite inventory tolerance" — wrong for prediction markets where the price slams to 0 or 1 in the last seconds.

Engineering choice:

- `horizon_seconds(t) = min(T_resolve − t, MAX_HORIZON)` with `MAX_HORIZON = 86400` (one day). Beyond a day, AS treats more inventory risk than is real (you can hedge / re-trade tomorrow).
- Below `CLOSE_OUT_THRESHOLD = 600s`, switch to **close-out mode**: replace `δ*` with an aggressive close-out spread that prices off the inventory directly:
  - if `q > 0`, lift the offer by `−ε` (tight ask, wide bid)
  - if `q < 0`, drop the bid by `+ε` (tight bid, wide ask)

This is implemented as a state-machine flag in `quoter.py`, not in the calibrator.

## 7. Quote-time post-processing

After computing `(p_bid, p_ask)`:

1. Round to tick (default $0.01).
2. Clip to `[tick, 1 − tick]`.
3. If `r ∈ [tick, 0.05]` or `r ∈ [0.95, 1 − tick]`, quote one-sided only — symmetric AS spreads near the boundary are negative-EV.
4. Optional cross-side sanity: the implied NO-side quote `(1 − p_ask, 1 − p_bid)` should not cross the actual NO market (Polymarket lists YES and NO as separate markets in some events; same outcome, sum to 1).

## 8. Bounds, validity, and what this DOESN'T solve

This bridge gives you AS-flavored quotes that respect Polymarket's data shape. It does **not**:

- Solve the boundary problem rigorously (a proper treatment would replace BM with a bounded diffusion like Jacobi process; the analytical AS no longer applies — that would be a research project, which the user said this isn't).
- Calibrate `γ` (no observable maps cleanly to risk aversion).
- Account for jump risk from news events. If you know an event is coming (debate, primary, FOMC), shrink `(T − t)` ad hoc, or pause quoting.
- Replace queue-position simulation. For a serious strategy, add a queue model on top of `λ̂(δ)`.

## 9. Validation strategy

Synthetic generator (`tests/test_calibrator.py`) injects:
- known logit-BM with `σ_truth`
- Poisson taker arrivals on each side with `A_truth, κ_truth`
- realistic book snapshot debouncing

Estimator should recover σ within ±25%, κ within ±30%, A within ±50% on a 5-minute simulated stream. These are loose tolerances appropriate for the noise level of these fits in production.

## 10. Files in this repo

- `polymarket_as/transforms.py` — logit / inverse, microprice
- `polymarket_as/state.py` — `OrderBookState`: ingests `book` and `trade` messages, exposes rolling buffer
- `polymarket_as/calibrator.py` — `ASCalibrator`: σ̂, (Â, κ̂) per side
- `polymarket_as/quoter.py` — `ASQuoter`: produces `(bid, ask)` from calibrator output + inventory
- `polymarket_as/api.py` — minimal REST + WSS client (HMAC signing, async iterator)
- `examples/calibrate_demo.py` — synthetic-data end-to-end demo (no API key)
- `tests/test_calibrator.py`, `tests/test_transforms.py`, `tests/test_quoter.py`
