"""
T212 Portfolio Checker — Daily Orchestrator

Daily workflow (budget: 19 Gemini requests):
  1. Fetch portfolio from T212, get live prices via yfinance, sync to Google Sheet
  2. Analyse watchlist symbols (if any filled by user)
  3. Market overview with VIX (1 request)
  4. Individual stock analysis with yfinance context (remaining budget, by weight)
  5. Stop at 19 requests, update sheet, wait for next day
"""

import os
import sys
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from fetch_portfolio import fetch_all_positions
from sheets import SheetManager
from market_data import get_batch_prices, get_vix, get_market_indices, build_stock_context
from analyse import (
    _get_client,
    AnalysisBudget,
    analyse_watchlist_symbol,
    analyse_market_overview,
    analyse_stock_advanced,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    log.info("=== T212 Portfolio Checker starting (daily run) ===")

    # ── Initialise services ────────────────────────────────────────────────
    gemini = _get_client()
    budget = AnalysisBudget()
    sheet = SheetManager()
    log.info("Sheet URL: %s", sheet.url)

    # ── Step 1: Fetch & sync portfolio with live prices ──────────────────
    log.info("Step 1 — Fetching portfolio from Trading 212...")
    positions = fetch_all_positions()
    if not positions:
        log.warning("No positions returned. Continuing with existing sheet data.")
    else:
        log.info("  %d positions fetched. Getting live prices via yfinance...", len(positions))
        symbols = [p.get("ticker", "") for p in positions if p.get("ticker")]
        live_prices = get_batch_prices(symbols)
        log.info("  Live prices fetched for %d symbols.", len(live_prices))
        sheet.sync_portfolio(positions, prices=live_prices)

    # ── Step 2: Analyse watchlist symbols ─────────────────────────────────
    log.info("Step 2 — Checking watchlist for symbols to analyse...")
    watchlist = sheet.get_watchlist()
    if watchlist:
        log.info("  %d watchlist symbols found.", len(watchlist))
        for item in watchlist:
            if budget.exhausted:
                log.info("  Budget exhausted. Remaining watchlist deferred to tomorrow.")
                break
            symbol = item["symbol"]
            log.info("  Analysing watchlist symbol: %s", symbol)
            if not budget.consume():
                break
            try:
                context = build_stock_context(symbol)
                analysis = analyse_watchlist_symbol(gemini, symbol, context)
                sheet.update_watchlist_analysis(symbol, analysis)
                log.info("  %s analysis written to sheet.", symbol)
            except Exception as e:
                log.error("  Failed to analyse %s: %s", symbol, e)
    else:
        log.info("  No watchlist symbols to analyse.")

    # ── Step 3: Market overview with VIX ─────────────────────────────────
    if not budget.exhausted:
        log.info("Step 3 — Running market overview with VIX...")
        if not budget.consume():
            log.info("  Budget exhausted. Market overview deferred.")
        else:
            try:
                market_data = get_market_indices()
                vix_data = get_vix()
                market_positions = positions or []

                analysis = analyse_market_overview(gemini, market_data, vix_data, market_positions)

                # Build structured entries for the sheet
                entries = []
                for name, data in market_data.items():
                    change = data.get("change_pct", 0)
                    entries.append({
                        "indicator": name,
                        "value": f"{data.get('price', 0):,.2f}",
                        "change_pct": f"{change:+.2f}%",
                        "sentiment": "Bullish" if change > 0.5 else "Bearish" if change < -0.5 else "Neutral",
                        "analysis": analysis if name == "S&P 500" else "",
                    })
                # Add VIX as its own row
                entries.append({
                    "indicator": "VIX",
                    "value": f"{vix_data.get('current', 0):.2f}",
                    "change_pct": f"{vix_data.get('change_pct', 0):+.1f}%",
                    "sentiment": vix_data.get("sentiment", "N/A"),
                    "analysis": "",
                })

                sheet.write_market_overview(entries)
                log.info("  Market overview written to sheet.")
            except Exception as e:
                log.error("  Market overview failed: %s", e)
    else:
        log.info("Step 3 — Skipped (budget exhausted).")

    # ── Step 4: Individual stock analysis ────────────────────────────────
    if not budget.exhausted:
        log.info("Step 4 — Running stock analysis (budget: %d remaining)...", budget.remaining)
        stocks = sheet.get_portfolio_for_analysis()
        analysed = 0
        for stock in stocks:
            if budget.exhausted:
                log.info("  Budget exhausted after %d stocks. Rest deferred to tomorrow.", analysed)
                break
            symbol = stock["symbol"]
            log.info("  Analysing %s (weight %.1s%%)...", symbol, stock.get("weight", "0"))
            if not budget.consume():
                break
            try:
                context = build_stock_context(symbol)
                analysis = analyse_stock_advanced(
                    gemini,
                    symbol=symbol,
                    financial_context=context,
                    amount=stock.get("amount", 0),
                    price=stock.get("price", 0),
                    weight=stock.get("weight", "0"),
                )
                sheet.update_portfolio_analysis(symbol, analysis)
                log.info("  %s analysis written to sheet.", symbol)
                analysed += 1
            except Exception as e:
                log.error("  Failed to analyse %s: %s", symbol, e)
    else:
        log.info("Step 4 — Skipped (budget exhausted).")

    # ── Summary ────────────────────────────────────────────────────────────
    log.info(
        "=== Pipeline complete — %d/%d Gemini requests used ===",
        budget.used, budget.max_requests,
    )
    log.info("Sheet: %s", sheet.url)


if __name__ == "__main__":
    main()
