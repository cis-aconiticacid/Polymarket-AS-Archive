"""Minimal Polymarket US REST + WebSocket client.

This module is structured so the calibrator's logic is fully decoupled from
the wire protocol — synthetic streams in tests bypass it entirely. The
real-API surface here is:

  PolymarketRESTClient.get_book(slug)        → BookSnapshot
  PolymarketRESTClient.get_bbo(slug)         → BookSnapshot (lite)
  PolymarketWSSClient.stream(slugs)          → async iterator over (BookSnapshot | TradeEvent)

Auth (per docs/api-reference/authentication, scraped):
  X-PM-Access-Key:  <key id>
  X-PM-Timestamp:   <ms since epoch>
  X-PM-Signature:   base64(HMAC_SHA256(secret, f"{timestamp}GET{path}"))

Requires `requests` for REST and `websockets` for WSS. Both pip-installable.
The synthetic demo and tests do NOT import this module, so absence of
`websockets` does not break the test suite.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional, Union

from polymarket_as.state import BookSnapshot, TradeEvent

REST_BASE = "https://gateway.polymarket.us"
WSS_URL = "wss://api.polymarket.us/v1/ws/markets"


@dataclass
class APICredentials:
    access_key: str
    secret: str           # base64-encoded HMAC secret
    passphrase: Optional[str] = None  # some endpoints require this; keep optional

    def sign(self, method: str, path: str, ts_ms: Optional[int] = None) -> dict:
        ts = str(ts_ms if ts_ms is not None else int(time.time() * 1000))
        msg = f"{ts}{method}{path}".encode()
        secret_bytes = base64.b64decode(self.secret) if _is_b64(self.secret) else self.secret.encode()
        sig = hmac.new(secret_bytes, msg, hashlib.sha256).digest()
        headers = {
            "X-PM-Access-Key": self.access_key,
            "X-PM-Timestamp": ts,
            "X-PM-Signature": base64.b64encode(sig).decode(),
        }
        if self.passphrase:
            headers["X-PM-Passphrase"] = self.passphrase
        return headers


def _is_b64(s: str) -> bool:
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


# -------- REST --------------------------------------------------------------

class PolymarketRESTClient:
    """Minimal REST client. Use sparingly — rate limit is 20 RPS public, 55/10s authenticated."""

    def __init__(self, credentials: Optional[APICredentials] = None, base_url: str = REST_BASE):
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        # Lazy-import requests so the module is loadable even if requests is missing
        import requests  # noqa: F401
        self._session = __import__("requests").Session()

    def _headers(self, method: str, path: str) -> dict:
        if self.credentials is None:
            return {}
        return self.credentials.sign(method, path)

    def get_book(self, slug: str) -> BookSnapshot:
        path = f"/v1/markets/{slug}/book"
        r = self._session.get(self.base_url + path, headers=self._headers("GET", path), timeout=10)
        r.raise_for_status()
        return _parse_book_response(r.json())

    def get_bbo(self, slug: str) -> BookSnapshot:
        path = f"/v1/markets/{slug}/bbo"
        r = self._session.get(self.base_url + path, headers=self._headers("GET", path), timeout=10)
        r.raise_for_status()
        return _parse_bbo_response(r.json())


def _parse_px(o) -> float:
    """API returns price as either {value, currency} where value may be float or string."""
    if isinstance(o, dict):
        v = o.get("value")
        return float(v)
    return float(o)


def _parse_book_response(payload: dict) -> BookSnapshot:
    md = payload["marketData"]
    bids = md.get("bids") or []
    offers = md.get("offers") or []
    if not bids or not offers:
        # Empty side — return a degenerate snapshot the state ingester will reject.
        bb = _parse_px(bids[0]["px"]) if bids else 0.0
        ba = _parse_px(offers[0]["px"]) if offers else 1.0
        bq = float(bids[0]["qty"]) if bids else 0.0
        aq = float(offers[0]["qty"]) if offers else 0.0
    else:
        bb = _parse_px(bids[0]["px"])
        ba = _parse_px(offers[0]["px"])
        bq = float(bids[0]["qty"])
        aq = float(offers[0]["qty"])
    ts_str = md.get("transactTime")
    ts = _parse_ts(ts_str) if ts_str else time.time()
    return BookSnapshot(
        ts=ts,
        best_bid=bb, best_ask=ba,
        bid_qty=bq, ask_qty=aq,
        state=md.get("state", "MARKET_STATE_OPEN"),
    )


def _parse_bbo_response(payload: dict) -> BookSnapshot:
    md = payload["marketData"]
    bb = _parse_px(md.get("bestBid", 0.0))
    ba = _parse_px(md.get("bestAsk", 1.0))
    bq = float(md.get("bidDepth", 0))
    aq = float(md.get("askDepth", 0))
    return BookSnapshot(
        ts=time.time(),
        best_bid=bb, best_ask=ba,
        bid_qty=bq, ask_qty=aq,
        state="MARKET_STATE_OPEN",
    )


def _parse_ts(s: str) -> float:
    """Parse RFC 3339 timestamp to epoch seconds. Cheap parser, no deps."""
    # Format: "2024-01-15T10:30:00Z" or with .ffffff
    import datetime as _dt
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return _dt.datetime.fromisoformat(s).timestamp()


# -------- WSS ---------------------------------------------------------------

class PolymarketWSSClient:
    """Async WSS client. Yields BookSnapshot or TradeEvent.

    Usage:
        async for msg in client.stream(["slug-a", "slug-b"]):
            state.ingest_snapshot(msg) if isinstance(msg, BookSnapshot) else state.ingest_trade(msg)
    """

    def __init__(
        self,
        credentials: APICredentials,
        url: str = WSS_URL,
        debounce: bool = True,
    ):
        self.credentials = credentials
        self.url = url
        self.debounce = debounce

    async def stream(self, slugs: Iterable[str]) -> AsyncIterator[Union[BookSnapshot, TradeEvent]]:
        # websockets is optional at module import time, required at use time.
        import websockets

        path = "/v1/ws/markets"
        headers = self.credentials.sign("GET", path)
        slug_list: List[str] = list(slugs)

        async with websockets.connect(self.url, extra_headers=list(headers.items())) as ws:
            # Subscribe to MARKET_DATA and TRADE for the given slugs.
            await ws.send(json.dumps({
                "subscribe": {
                    "requestId": "md-1",
                    "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
                    "marketSlugs": slug_list,
                    "responsesDebounced": self.debounce,
                }
            }))
            await ws.send(json.dumps({
                "subscribe": {
                    "requestId": "trade-1",
                    "subscriptionType": "SUBSCRIPTION_TYPE_TRADE",
                    "marketSlugs": slug_list,
                }
            }))
            async for raw in ws:
                msg = json.loads(raw)
                if "heartbeat" in msg:
                    continue
                if "marketData" in msg:
                    snap = _wss_to_book(msg["marketData"])
                    if snap is not None:
                        yield snap
                elif "trade" in msg:
                    ev = _wss_to_trade(msg["trade"])
                    if ev is not None:
                        yield ev
                elif "error" in msg:
                    raise RuntimeError(f"WSS subscription error: {msg['error']}")


def _wss_to_book(md: dict) -> Optional[BookSnapshot]:
    bids = md.get("bids") or []
    offers = md.get("offers") or []
    if not bids or not offers:
        return None
    ts = _parse_ts(md["transactTime"]) if md.get("transactTime") else time.time()
    return BookSnapshot(
        ts=ts,
        best_bid=_parse_px(bids[0]["px"]),
        best_ask=_parse_px(offers[0]["px"]),
        bid_qty=float(bids[0]["qty"]),
        ask_qty=float(offers[0]["qty"]),
        state=md.get("state", "MARKET_STATE_OPEN"),
    )


def _wss_to_trade(tr: dict) -> Optional[TradeEvent]:
    taker = tr.get("taker") or {}
    side = taker.get("side", "")
    if "BUY" in side:
        ts_side = "BUY"
    elif "SELL" in side:
        ts_side = "SELL"
    else:
        return None
    ts = _parse_ts(tr["tradeTime"]) if tr.get("tradeTime") else time.time()
    return TradeEvent(
        ts=ts,
        price=_parse_px(tr["price"]),
        qty=float(tr["quantity"]["value"]) if isinstance(tr.get("quantity"), dict) else float(tr["quantity"]),
        taker_side=ts_side,
    )
