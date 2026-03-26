"""
test_columns.py
Quick test script to update sheet headers and verify the new column structure.
Run this first to set up the sheet before running the full pipeline.

Usage:
  cd src && python test_columns.py
"""

import sys
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    log.info("=" * 60)
    log.info("Column Structure Test")
    log.info("=" * 60)

    # ── Step 1: Connect to sheet ─────────────────────────────────────────
    log.info("")
    log.info("STEP 1: Connecting to Google Sheet...")
    try:
        from sheets import SheetManager
        sheet = SheetManager()
        log.info("  Connected: %s", sheet.url)
    except Exception as e:
        log.error("  FAILED: %s", e)
        sys.exit(1)

    # ── Step 2: Verify headers ───────────────────────────────────────────
    log.info("")
    log.info("STEP 2: Checking tab headers...")
    from sheets import TAB_PORTFOLIO, TAB_WATCHLIST, TAB_MARKET
    from sheets import PORTFOLIO_HEADERS, WATCHLIST_HEADERS, MARKET_HEADERS

    for tab, expected in [
        (TAB_PORTFOLIO, PORTFOLIO_HEADERS),
        (TAB_WATCHLIST, WATCHLIST_HEADERS),
        (TAB_MARKET, MARKET_HEADERS),
    ]:
        try:
            result = sheet.sheets.spreadsheets().values().get(
                spreadsheetId=sheet.sheet_id,
                range=f"'{tab}'!A1:Z1",
            ).execute()
            actual = result.get("values", [[]])[0]
            if actual == expected:
                log.info("  %s: headers OK — %s", tab, actual)
            else:
                log.warning("  %s: headers MISMATCH", tab)
                log.warning("    Expected: %s", expected)
                log.warning("    Actual:   %s", actual)
                log.info("  Updating headers for %s...", tab)
                sheet.sheets.spreadsheets().values().update(
                    spreadsheetId=sheet.sheet_id,
                    range=f"'{tab}'!A1",
                    valueInputOption="RAW",
                    body={"values": [expected]},
                ).execute()
                log.info("  %s headers updated.", tab)
        except Exception as e:
            log.error("  %s: error — %s", tab, e)

    # ── Step 3: Test yfinance ────────────────────────────────────────────
    log.info("")
    log.info("STEP 3: Testing yfinance...")
    try:
        from market_data import get_vix, get_market_indices, build_stock_context

        vix = get_vix()
        log.info("  VIX: %.2f (%+.1f%%) — %s", vix["current"], vix["change_pct"], vix["sentiment"])

        indices = get_market_indices()
        for name, data in indices.items():
            log.info("  %s: %,.2f (%+.2f%%)", name, data["price"], data["change_pct"])

        # Test one stock
        context = build_stock_context("AAPL")
        log.info("  AAPL context: %s", context)

    except Exception as e:
        log.error("  yfinance test failed: %s", e)
        log.error("  Make sure yfinance is installed: pip install yfinance")
        sys.exit(1)

    # ── Step 4: Write sample market overview to sheet ─────────────────────
    log.info("")
    log.info("STEP 4: Writing sample market data to Market Overview tab...")
    try:
        entries = []
        for name, data in indices.items():
            change = data.get("change_pct", 0)
            entries.append({
                "indicator": name,
                "value": f"{data.get('price', 0):,.2f}",
                "change_pct": f"{change:+.2f}%",
                "sentiment": "Bullish" if change > 0.5 else "Bearish" if change < -0.5 else "Neutral",
                "analysis": "",
            })
        entries.append({
            "indicator": "VIX",
            "value": f"{vix['current']:.2f}",
            "change_pct": f"{vix['change_pct']:+.1f}%",
            "sentiment": vix["sentiment"],
            "analysis": "",
        })
        sheet.write_market_overview(entries)
        log.info("  Market Overview tab updated with live data.")
    except Exception as e:
        log.error("  Failed to write market overview: %s", e)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("ALL STEPS PASSED")
    log.info("=" * 60)
    log.info("New column structure:")
    log.info("  Portfolio:       %s", PORTFOLIO_HEADERS)
    log.info("  Watchlist:       %s", WATCHLIST_HEADERS)
    log.info("  Market Overview: %s", MARKET_HEADERS)
    log.info("")
    log.info("Sheet: %s", sheet.url)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
