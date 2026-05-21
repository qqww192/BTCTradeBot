"""
Gemini AI weekly optimiser — runs Sunday 23:00 UTC via crontab.

Steps
-----
1. Reads the last 7 days of trade data.
2. Computes performance metrics (win rate, Sharpe, fee drag, etc.).
3. Sends metrics + regime history to Gemini AI.
4. Gemini returns proposed new grid parameters as JSON.
5. Walk-forward simulation: applies proposed params to last 30 days
   of price data and checks if net return improves on current params.
6. If confirmed: writes new params to config/grid_params.json.
7. Sends Telegram summary regardless of outcome.

Overfitting guard
-----------------
Gemini is explicitly instructed not to chase last week's noise.
Walk-forward validation on 30 days (not 7) is required before any
param change is accepted.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trading.trade_logger  import read_all, read_since
from trading.cdx_client    import CDXClient, CDXError

CONFIG_FILE = ROOT / "config" / "grid_params.json"
REGIME_FILE = ROOT / "data"   / "regime.json"
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


# ------------------------------------------------------------------ #
#  Performance metrics                                                 #
# ------------------------------------------------------------------ #

SAFE_BOUNDS = {
    "spacing_pct": (0.55, 3.0),
    "range_pct":   (2.0,  15.0),
    "levels":      (4,    20),
    "capital_pct": (0.40, 0.80),
    "kill_pct":    (0.05, 0.15),
}


def validate_params(proposed: dict) -> list[str]:
    """Return a list of violation strings; empty list means params are safe."""
    violations = []
    for key, (lo, hi) in SAFE_BOUNDS.items():
        val = proposed.get(key)
        if val is None:
            violations.append(f"{key} missing")
        elif not (lo <= val <= hi):
            violations.append(f"{key}={val} outside [{lo}, {hi}]")
    return violations


def compute_metrics(trades: list[dict]) -> dict:
    sells = [t for t in trades if t["side"] == "SELL"]
    if not sells:
        return {
            "trades_total": len(trades), "sells": 0, "win_rate_pct": 0,
            "avg_net_gbp": 0, "total_net_gbp": 0, "fee_drag_pct": 0,
            "sharpe": 0, "max_loss_gbp": 0, "avg_win_gbp": 0,
            "note": "no_sells_this_week",
        }

    nets     = [t["net_gbp"] for t in sells]
    wins     = [n for n in nets if n > 0]
    losses   = [n for n in nets if n <= 0]
    gross    = sum(t["gross_gbp"] for t in trades)
    fees_gbp = sum(t["fee_usdt"]  for t in trades) / float(os.environ.get("GBP_USD_RATE", "1.27"))
    net_tot  = sum(nets)
    fee_drag = (fees_gbp / gross * 100) if gross else 0

    mean_ret = net_tot / len(nets) if nets else 0
    std_ret  = math.sqrt(sum((n - mean_ret) ** 2 for n in nets) / len(nets)) if len(nets) > 1 else 0
    sharpe   = mean_ret / std_ret if std_ret else 0

    return {
        "trades_total":   len(trades),
        "sells":          len(sells),
        "win_rate_pct":   round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "avg_net_gbp":    round(mean_ret, 4),
        "total_net_gbp":  round(net_tot, 2),
        "fee_drag_pct":   round(fee_drag, 1),
        "sharpe":         round(sharpe, 2),
        "max_loss_gbp":   round(min(losses, default=0), 4),
        "avg_win_gbp":    round(sum(wins) / len(wins), 4) if wins else 0,
    }


# ------------------------------------------------------------------ #
#  Gemini call                                                         #
# ------------------------------------------------------------------ #

def ask_gemini(metrics_7d: dict, current_config: dict, regime: str) -> dict | None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[optimiser] GEMINI_API_KEY not set — skipping AI review.")
        return None

    prompt = f"""
You are a conservative crypto trading bot optimiser.
The bot uses a spot grid strategy on BTC/USDT on crypto.com Exchange.
It has a £150 total capital pool and a 10% weekly kill switch.

Current grid parameters:
{json.dumps(current_config, indent=2)}

Current market regime: {regime}

Last 7 days performance metrics:
{json.dumps(metrics_7d, indent=2)}

Your task: propose new grid parameters for next week.

Rules you must follow:
- Prioritise capital preservation over return maximisation.
- Do NOT chase last week's results — think about what is robust over 30 days.
- If the current params are already performing well (win_rate > 60%, fee_drag < 30%), 
  keep changes minimal or return the same params.
- Tighten the kill switch threshold (kill_pct) if trending_dn regime.
- spacing_pct must always be > 0.55 (minimum to beat 0.25% maker fee × 2).
- levels must be between 6 and 16.
- capital_pct must be between 0.50 and 0.80.
- range_pct must be between 3.0 and 10.0.

Return ONLY a valid JSON object with these exact keys:
{{
  "instrument":   "BTC_USDT",
  "spacing_pct":  <float>,
  "range_pct":    <float>,
  "levels":       <int>,
  "capital_pct":  <float>,
  "total_capital": {current_config.get("total_capital", 150)},
  "gbp_usd_rate": {current_config.get("gbp_usd_rate", 1.27)},
  "kill_pct":     <float>,
  "rationale":    "<one sentence>"
}}
No other text. No markdown. No code blocks. Pure JSON only.
"""

    try:
        resp = httpx.post(
            f"{GEMINI_URL}?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Strip accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[optimiser] Gemini call failed: {e}")
        return None


# ------------------------------------------------------------------ #
#  Walk-forward simulation                                             #
# ------------------------------------------------------------------ #

def simulate_return(candles: list[dict], config: dict) -> float:
    """
    Simplified walk-forward simulation.
    Counts how many grid-spacing moves fit within each day's range,
    applies fee, and returns the total estimated net return %.
    """
    spacing_pct = config["spacing_pct"] / 100
    fee_pct     = 0.0025   # 0.25% maker per leg × 2 legs per round trip

    total_return = 0.0
    for c in candles:
        daily_range_pct = (c["high"] - c["low"]) / c["close"]
        # Estimate fills per day = range / spacing, capped at levels / 2
        fills = min(daily_range_pct / spacing_pct, config["levels"] / 2)
        gross = fills * spacing_pct * config["capital_pct"]
        fees  = fills * fee_pct     * 2
        total_return += gross - fees

    return round(total_return * 100, 2)  # as percentage


def walk_forward_confirms(
    candles: list[dict],
    current: dict,
    proposed: dict,
) -> tuple[bool, float, float]:
    """
    Return (accepted, current_return, proposed_return).
    Accept if proposed return > current return.
    """
    curr_ret = simulate_return(candles, current)
    prop_ret = simulate_return(candles, proposed)
    return prop_ret > curr_ret, curr_ret, prop_ret


# ------------------------------------------------------------------ #
#  Telegram                                                            #
# ------------------------------------------------------------------ #

def send_telegram(msg: str) -> None:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(f"[optimiser] Telegram: {msg}")
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as e:
        print(f"[optimiser] Telegram send failed: {e}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def run() -> None:
    print("[optimiser] Starting weekly Gemini optimisation...")

    # Load current config
    current_config = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}

    # Load regime
    regime = "ranging"
    if REGIME_FILE.exists():
        regime = json.loads(REGIME_FILE.read_text()).get("regime", "ranging")

    # Compute last-7-day metrics
    since_7d  = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    trades_7d = read_since(since_7d)
    metrics   = compute_metrics(trades_7d)

    print(f"[optimiser] 7-day metrics: {metrics}")

    # Fetch 30 days of candles for walk-forward
    cdx     = CDXClient()
    candles = cdx.get_candlesticks("BTC_USDT", timeframe="1D", count=30)

    # Ask Gemini
    proposed = ask_gemini(metrics, current_config, regime)

    if proposed is None:
        send_telegram(
            "🔄 *Weekly optimisation*\n"
            "Gemini AI unavailable — keeping current parameters.\n"
            f"7-day net P&L: £{metrics.get('total_net_gbp', 0):.2f}"
        )
        return

    rationale = proposed.pop("rationale", "No rationale provided.")

    # Safety gate — reject any params outside hard bounds before walk-forward
    violations = validate_params(proposed)
    if violations:
        msg = "⚠️ *Gemini proposed unsafe params — rejected*\n" + "\n".join(violations)
        send_telegram(msg)
        print(f"[optimiser] Unsafe proposal rejected: {violations}")
        return

    # Walk-forward validation
    accepted, curr_ret, prop_ret = walk_forward_confirms(candles, current_config, proposed)

    if accepted:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(proposed, indent=2))
        action  = "✅ Parameters updated"
        details = (
            f"Old spacing: {current_config.get('spacing_pct')}% → New: {proposed.get('spacing_pct')}%\n"
            f"Old range: ±{current_config.get('range_pct')}% → New: ±{proposed.get('range_pct')}%\n"
            f"Old levels: {current_config.get('levels')} → New: {proposed.get('levels')}\n"
            f"Walk-forward: {curr_ret:.2f}% → {prop_ret:.2f}%"
        )
    else:
        action  = "⏸ Parameters unchanged (walk-forward did not confirm improvement)"
        details = f"Walk-forward: current {curr_ret:.2f}% vs proposed {prop_ret:.2f}%"

    send_telegram(
        f"🔄 *Weekly Gemini optimisation*\n"
        f"{action}\n\n"
        f"_{rationale}_\n\n"
        f"{details}\n\n"
        f"7-day: {metrics.get('trades_total', 0)} trades · "
        f"win rate {metrics.get('win_rate_pct', 0)}% · "
        f"net £{metrics.get('total_net_gbp', 0):.2f}\n"
        f"New week starts now. P&L counter reset."
    )
    print(f"[optimiser] Done. {action}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    run()
