"""
Regime classifier — run every 4 hours via crontab.

Fetches 30 daily candles from crypto.com, computes ATR-14 and
Bollinger Band Width, classifies the market, and writes the result
to data/regime.json. The grid_trader reads this file to adjust
its spacing and range before placing orders.

Regimes:
  ranging     — low volatility, tight bands → tighter grid (0.6–0.8%)
  trending_up — sustained upward ATR spike  → widen grid ceiling, reduce buys
  trending_dn — sustained downward ATR spike → tighten kill switch threshold
  volatile    — high ATR + wide BBW          → widen grid (1.2–1.5%)
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trading.cdx_client import CDXClient


DATA_FILE   = ROOT / "data" / "regime.json"
INSTRUMENT  = "BTC_USDT"


# ------------------------------------------------------------------ #
#  Indicator helpers                                                   #
# ------------------------------------------------------------------ #

def compute_atr(candles: list[dict], period: int = 14) -> float:
    """
    Average True Range over `period` candles.
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    trs = []
    for i in range(1, len(candles)):
        high  = candles[i]["high"]
        low   = candles[i]["low"]
        prev  = candles[i - 1]["close"]
        tr    = max(high - low, abs(high - prev), abs(low - prev))
        trs.append(tr)
    if len(trs) < period:
        return float("nan")
    # Simple moving average of TR over last `period` values
    return sum(trs[-period:]) / period


def compute_bbw(candles: list[dict], period: int = 20) -> float:
    """
    Bollinger Band Width = (upper - lower) / middle
    Uses closing prices over `period` candles.
    """
    closes = [c["close"] for c in candles[-period:]]
    if len(closes) < period:
        return float("nan")
    mean   = sum(closes) / period
    stddev = math.sqrt(sum((x - mean) ** 2 for x in closes) / period)
    upper  = mean + 2 * stddev
    lower  = mean - 2 * stddev
    return (upper - lower) / mean if mean else float("nan")


def classify(atr: float, bbw: float, candles: list[dict]) -> str:
    """
    Classify market regime from ATR and BBW.

    Thresholds are expressed as percentages of current price so they
    scale with BTC's absolute level.
    """
    price       = candles[-1]["close"]
    atr_pct     = (atr / price) * 100 if price else 0
    # BBW is already normalised (fraction of price)
    bbw_pct     = bbw * 100

    # Detect trend direction via 5-candle slope
    recent      = [c["close"] for c in candles[-5:]]
    slope       = (recent[-1] - recent[0]) / recent[0] * 100

    if bbw_pct < 3.0 and atr_pct < 1.5:
        return "ranging"
    elif bbw_pct > 6.0 or atr_pct > 3.5:
        return "volatile"
    elif slope > 4.0:
        return "trending_up"
    elif slope < -4.0:
        return "trending_dn"
    else:
        return "ranging"


# ------------------------------------------------------------------ #
#  Recommended grid params per regime                                  #
# ------------------------------------------------------------------ #

REGIME_PARAMS = {
    "ranging": {
        "spacing_pct":   0.8,
        "range_pct":     5.0,
        "levels":        10,
        "capital_pct":   0.70,
        "kill_pct":      0.10,
    },
    "trending_up": {
        "spacing_pct":   1.2,
        "range_pct":     7.0,
        "levels":        8,
        "capital_pct":   0.60,   # lighter — trend may break grid ceiling
        "kill_pct":      0.10,
    },
    "trending_dn": {
        "spacing_pct":   1.0,
        "range_pct":     5.0,
        "levels":        8,
        "capital_pct":   0.55,   # lightest — protect capital in downtrend
        "kill_pct":      0.08,   # tighter kill switch
    },
    "volatile": {
        "spacing_pct":   1.4,
        "range_pct":     8.0,
        "levels":        10,
        "capital_pct":   0.65,
        "kill_pct":      0.10,
    },
}


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def run() -> None:
    cdx     = CDXClient()
    candles = cdx.get_candlesticks(INSTRUMENT, timeframe="1D", count=30)

    if len(candles) < 20:
        print("Not enough candle data — skipping regime update.")
        return

    atr     = compute_atr(candles)
    bbw     = compute_bbw(candles)
    regime  = classify(atr, bbw, candles)
    params  = REGIME_PARAMS[regime]

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps(
            {
                "regime":        regime,
                "atr":           round(atr, 2),
                "bbw_pct":       round(bbw * 100, 2),
                "recommended":   params,
                "updated_at":    datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    print(f"[regime] {regime} | ATR={atr:.0f} | BBW={bbw*100:.1f}% | "
          f"spacing={params['spacing_pct']}% | range=±{params['range_pct']}%")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    run()
