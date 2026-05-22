"""
Risk manager — enforces the weekly kill switch and tracks P&L.

State is stored in data/weekly_state.json.  This file is the single
source of truth for whether the bot is allowed to trade this week.

Kill switch logic
-----------------
  - Weekly P&L is computed as the sum of all net gains/losses since
    Monday 00:00 UTC.
  - If weekly_pnl <= -(total_capital * kill_pct), trading halts.
  - The state resets automatically on Monday 00:00 UTC.
  - A Telegram alert is sent when the switch triggers.

The grid_trader calls `is_kill_switch_active()` at the start of
every 5-minute run.  If it returns True, the run exits immediately
after cancelling any remaining open orders.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT       = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / "data" / "weekly_state.json"

# Capital in GBP — loaded from env so it's easy to change
TOTAL_CAPITAL_GBP = float(os.environ.get("TOTAL_CAPITAL_GBP", "150"))


def _monday_utc() -> datetime:
    """Return the most recent Monday 00:00:00 UTC."""
    now   = datetime.now(timezone.utc)
    delta = timedelta(days=now.weekday())
    return (now - delta).replace(hour=0, minute=0, second=0, microsecond=0)


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _reset_if_new_week(state: dict) -> dict:
    """
    If the stored week_start is before this Monday, reset the state.
    This handles the automatic Monday reset without any cron job.
    """
    monday = _monday_utc().isoformat()
    if state.get("week_start") != monday:
        state = {
            "week_start":       monday,
            "weekly_pnl_gbp":   0.0,
            "kill_switch_on":   False,
            "kill_trigger_at":  None,
            "trades_this_week": 0,
        }
        _save(state)
    return state


def get_state() -> dict:
    """Load, reset-if-needed, and return the current weekly state."""
    return _reset_if_new_week(_load())


def is_kill_switch_active() -> bool:
    """Return True if trading should be halted this week."""
    return get_state().get("kill_switch_on", False)


def record_trade(net_pnl_gbp: float) -> dict:
    """
    Add a completed trade's net P&L (after fees, in GBP) to the
    weekly total and check whether the kill switch should fire.

    Returns the updated state.
    """
    state               = get_state()
    state["weekly_pnl_gbp"]   += net_pnl_gbp
    state["trades_this_week"] += 1

    kill_pct      = float(os.environ.get("KILL_SWITCH_PCT", "0.10"))
    kill_threshold = -(TOTAL_CAPITAL_GBP * kill_pct)

    if state["weekly_pnl_gbp"] <= kill_threshold and not state["kill_switch_on"]:
        state["kill_switch_on"]  = True
        state["kill_trigger_at"] = datetime.now(timezone.utc).isoformat()
        _save(state)
        _send_kill_alert(state)
    else:
        _save(state)

    return state


def record_warning(state: dict) -> None:
    """
    Send a Telegram warning when P&L hits the 50%-of-kill threshold.
    Called by grid_trader — does nothing if warning already sent this week.
    """
    if state.get("warning_sent"):
        return
    kill_pct       = float(os.environ.get("KILL_SWITCH_PCT", "0.10"))
    warning_thresh = -(TOTAL_CAPITAL_GBP * kill_pct * 0.5)
    if state["weekly_pnl_gbp"] <= warning_thresh:
        _send_telegram(
            f"⚠️ *Grid bot warning*\n"
            f"Weekly P&L: £{state['weekly_pnl_gbp']:.2f} "
            f"(50% of kill switch threshold)\n"
            f"Kill switch triggers at -£{abs(TOTAL_CAPITAL_GBP * kill_pct):.2f}"
        )
        state["warning_sent"] = True
        _save(state)


def _send_kill_alert(state: dict) -> None:
    kill_pct = float(os.environ.get("KILL_SWITCH_PCT", "0.10"))
    _send_telegram(
        f"🛑 *Kill switch triggered*\n"
        f"Weekly P&L reached £{state['weekly_pnl_gbp']:.2f} "
        f"(limit: -£{TOTAL_CAPITAL_GBP * kill_pct:.2f})\n"
        f"Trading paused until Monday 00:00 UTC.\n"
        f"Trades this week: {state['trades_this_week']}"
    )


def _send_telegram(message: str) -> None:
    """Fire-and-forget Telegram message. Errors are logged, not raised."""
    import httpx
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(f"[risk] Telegram not configured — message: {message}")
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as exc:
        print(f"[risk] Telegram send failed: {exc}")
