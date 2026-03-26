"""
market_data.py
Fetch live prices and key financial metrics via yfinance.
"""

import logging
from typing import Any

import yfinance as yf

log = logging.getLogger(__name__)


def get_stock_info(symbol: str) -> dict[str, Any]:
    """
    Fetch current price and key financials for a single ticker.
    Returns a dict with price, pe, market_cap, 52w_high, 52w_low, etc.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "prev_close": info.get("previousClose", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "short_name": info.get("shortName", symbol),
        }
    except Exception as e:
        log.warning("Failed to fetch yfinance data for %s: %s", symbol, e)
        return {"symbol": symbol, "price": 0}


def get_batch_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for multiple tickers in one call."""
    if not symbols:
        return {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
        prices = {}
        for sym in symbols:
            try:
                info = tickers.tickers[sym].info
                prices[sym] = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            except Exception:
                prices[sym] = 0
        return prices
    except Exception as e:
        log.warning("Batch price fetch failed: %s", e)
        return {s: 0 for s in symbols}


def get_vix() -> dict[str, Any]:
    """Fetch current VIX (CBOE Volatility Index) data."""
    try:
        vix = yf.Ticker("^VIX")
        info = vix.info
        hist = vix.history(period="5d")
        current = info.get("regularMarketPrice", 0)
        prev = info.get("previousClose", 0)

        # Determine sentiment
        if current < 15:
            sentiment = "Low volatility (complacent)"
        elif current < 20:
            sentiment = "Normal"
        elif current < 30:
            sentiment = "Elevated (cautious)"
        else:
            sentiment = "High fear (extreme caution)"

        return {
            "current": current,
            "prev_close": prev,
            "change_pct": ((current - prev) / prev * 100) if prev else 0,
            "sentiment": sentiment,
        }
    except Exception as e:
        log.warning("Failed to fetch VIX: %s", e)
        return {"current": 0, "prev_close": 0, "change_pct": 0, "sentiment": "N/A"}


def get_market_indices() -> dict[str, dict]:
    """Fetch key market indices for overview."""
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "FTSE 100": "^FTSE",
        "VIX": "^VIX",
    }
    results = {}
    for name, sym in indices.items():
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get("regularMarketPrice", 0)
            prev = info.get("previousClose", 0)
            results[name] = {
                "price": price,
                "prev_close": prev,
                "change_pct": ((price - prev) / prev * 100) if prev else 0,
            }
        except Exception as e:
            log.warning("Failed to fetch %s: %s", name, e)
            results[name] = {"price": 0, "prev_close": 0, "change_pct": 0}
    return results


def build_stock_context(symbol: str) -> str:
    """Build a concise financial context string for the AI prompt."""
    info = get_stock_info(symbol)
    parts = [f"Price: ${info.get('price', 0):.2f}"]

    if info.get("pe_ratio"):
        parts.append(f"P/E: {info['pe_ratio']:.1f}")
    if info.get("forward_pe"):
        parts.append(f"Fwd P/E: {info['forward_pe']:.1f}")
    if info.get("market_cap"):
        mc = info["market_cap"]
        if mc >= 1e12:
            parts.append(f"MCap: ${mc/1e12:.1f}T")
        elif mc >= 1e9:
            parts.append(f"MCap: ${mc/1e9:.1f}B")
        else:
            parts.append(f"MCap: ${mc/1e6:.0f}M")
    if info.get("52w_high"):
        parts.append(f"52w: ${info['52w_low']:.2f}-${info['52w_high']:.2f}")
    if info.get("revenue_growth") is not None:
        parts.append(f"Rev Growth: {info['revenue_growth']*100:.1f}%")
    if info.get("profit_margin") is not None:
        parts.append(f"Margin: {info['profit_margin']*100:.1f}%")
    if info.get("debt_to_equity") is not None:
        parts.append(f"D/E: {info['debt_to_equity']:.1f}")
    if info.get("beta") is not None:
        parts.append(f"Beta: {info['beta']:.2f}")

    return " | ".join(parts)
