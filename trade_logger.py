"""
Trade logger — append-only JSON ledger stored in data/trades.json.

Every filled order is appended as one line to make the file easy to
tail, parse, and back up.  The daily reporter and Gemini optimiser
read this file to compute statistics.

Each entry:
{
  "ts":          "2026-05-21T09:12:34+00:00",   # fill timestamp (UTC ISO)
  "order_id":    "12345678",
  "side":        "BUY" | "SELL",
  "price_usdt":  103240.50,                      # fill price
  "qty_btc":     0.000965,                       # BTC quantity
  "fee_usdt":    0.257,                          # fee paid
  "fee_pct":     0.0025,                         # fee as fraction of value
  "gross_gbp":   0.00,                           # gross gain (SELL only; BUY = 0)
  "net_gbp":     0.00,                           # net after fees (SELL only; BUY = negative fees)
  "regime":      "ranging",
  "grid_level":  5,
  "week_start":  "2026-05-18T00:00:00+00:00"
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT       = Path(__file__).resolve().parents[2]
TRADE_FILE = ROOT / "data" / "trades.json"

# Approximate GBP/USD rate — updated weekly by the Gemini optimiser.
# If env var GBP_USD_RATE is set, that takes precedence.
import os
GBP_USD_RATE = float(os.environ.get("GBP_USD_RATE", "1.27"))


def append(
    order_id:    str,
    side:        str,
    price_usdt:  float,
    qty_btc:     float,
    fee_usdt:    float,
    regime:      str,
    grid_level:  int,
    week_start:  str,
    buy_price_usdt: Optional[float] = None,   # needed to compute SELL gross
) -> dict:
    """
    Append one filled trade to the ledger.

    For BUY orders, gross_gbp = 0 (cost, not profit).
    For SELL orders, gross_gbp = (fill_price - buy_price) * qty / GBP_USD_RATE.
    net_gbp = gross_gbp - (fee_usdt / GBP_USD_RATE).
    """
    value_usdt = price_usdt * qty_btc

    if side.upper() == "SELL" and buy_price_usdt is not None:
        gross_usdt = (price_usdt - buy_price_usdt) * qty_btc
    else:
        gross_usdt = 0.0

    gross_gbp = gross_usdt / GBP_USD_RATE
    fee_gbp   = fee_usdt   / GBP_USD_RATE
    net_gbp   = gross_gbp - fee_gbp

    entry = {
        "ts":           datetime.now(timezone.utc).isoformat(),
        "order_id":     order_id,
        "side":         side.upper(),
        "price_usdt":   round(price_usdt, 2),
        "qty_btc":      round(qty_btc, 8),
        "value_usdt":   round(value_usdt, 4),
        "fee_usdt":     round(fee_usdt, 6),
        "fee_pct":      round(fee_usdt / value_usdt, 6) if value_usdt else 0,
        "gross_gbp":    round(gross_gbp, 4),
        "net_gbp":      round(net_gbp, 4),
        "regime":       regime,
        "grid_level":   grid_level,
        "week_start":   week_start,
    }

    TRADE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_all() -> list[dict]:
    """Return all trades from the ledger."""
    if not TRADE_FILE.exists():
        return []
    trades = []
    for line in TRADE_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return trades


def read_since(iso_ts: str) -> list[dict]:
    """Return trades with ts >= iso_ts."""
    return [t for t in read_all() if t["ts"] >= iso_ts]


def read_yesterday() -> list[dict]:
    """Return trades from the previous UTC day."""
    from datetime import timedelta
    now       = datetime.now(timezone.utc)
    yesterday = (now - timedelta(days=1)).date().isoformat()
    today     = now.date().isoformat()
    return [t for t in read_all() if yesterday <= t["ts"][:10] < today]


def read_this_week(week_start: str) -> list[dict]:
    """Return all trades for the current week."""
    return [t for t in read_all() if t.get("week_start") == week_start]
