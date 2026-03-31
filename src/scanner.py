"""
scanner.py
Stock scanner — screens a universe of US and LSE stocks against
user-defined conditions from the Scanner sheet tab.

Uses yfinance only (zero Gemini calls).

Two-pass filtering:
  Pass 1: Filter on .info-based metrics (P/E, market cap, growth, etc.)
  Pass 2: For survivors, fetch price history for computed metrics (RSI, DMA)
"""

import logging
import time
from typing import Any

import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)

# ── Stock Universes ──────────────────────────────────────────────────────────
# yfinance tickers. LSE stocks use .L suffix.

US_MEGA50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "XOM", "V", "MA", "PG", "JNJ", "COST", "HD", "ABBV",
    "WMT", "NFLX", "KO", "MRK", "CRM", "BAC", "CVX", "PEP", "AMD", "TMO",
    "LIN", "ADBE", "ORCL", "ACN", "MCD", "CSCO", "ABT", "WFC", "GE", "DHR",
    "TXN", "PM", "QCOM", "INTU", "ISRG", "AMGN", "CAT", "NOW", "IBM", "GS",
]

US_LARGE100 = US_MEGA50 + [
    "RTX", "VZ", "AMAT", "NEE", "LOW", "SPGI", "BLK", "HON", "UNP", "DE",
    "PFE", "T", "ADP", "BKNG", "MDLZ", "SCHW", "CI", "LMT", "MMC", "SYK",
    "GILD", "CB", "ADI", "LRCX", "TMUS", "SO", "ZTS", "DUK", "MO", "CL",
    "REGN", "CME", "BSX", "PLD", "ICE", "PYPL", "SNPS", "CDNS", "EQIX", "AON",
    "MCK", "ITW", "SHW", "ABNB", "PGR", "KLAC", "APH", "MSI", "WELL", "CRWD",
]

US_SP500_EXTENDED = US_LARGE100 + [
    "EMR", "CTAS", "ORLY", "ADSK", "MAR", "GM", "F", "ROP", "MCHP", "NXPI",
    "HCA", "AJG", "TT", "PSA", "FTNT", "CARR", "OKE", "MNST", "AEP", "AIG",
    "FAST", "DLR", "SRE", "ALL", "CCI", "PAYX", "KMB", "MSCI", "GIS", "EA",
    "ROST", "TEL", "D", "HLT", "VRSK", "BK", "WEC", "YUM", "CTSH", "DXCM",
    "PCAR", "STZ", "EXC", "HPQ", "IDXX", "CPRT", "ODFL", "ON", "CSGP", "GEHC",
    "KDP", "AWK", "WBD", "DOV", "GPN", "ANSS", "KEYS", "CDW", "MPWR", "MTD",
    "EFX", "WAT", "BR", "VLTO", "FSLR", "WST", "TRGP", "RCL", "CCL", "DAL",
    "UAL", "LUV", "ALB", "FMC", "ENPH", "SEDG", "PLUG", "RIVN", "LCID", "NIO",
    "SOFI", "PLTR", "SNOW", "DDOG", "NET", "ZS", "PANW", "OKTA", "HUBS", "BILL",
    "SQ", "SHOP", "MELI", "SE", "GRAB", "NU", "COIN", "HOOD", "RBLX", "U",
]

LSE_FTSE100 = [
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "BATS.L",
    "DGE.L", "LSEG.L", "REL.L", "NG.L", "CRH.L", "AHT.L", "AAL.L", "GLEN.L",
    "RKT.L", "CPG.L", "MNG.L", "PRU.L", "SSE.L", "VOD.L", "BA.L", "LLOY.L",
    "BARC.L", "STAN.L", "NWG.L", "ANTO.L", "ABF.L", "BT-A.L", "WPP.L", "III.L",
    "TSCO.L", "SGRO.L", "PSON.L", "INF.L", "IMB.L", "AV.L", "LGEN.L", "HLMA.L",
    "SVT.L", "UU.L", "ADM.L", "SN.L", "SDR.L", "RR.L", "EXPN.L", "IHG.L",
    "BNZL.L", "WEIR.L", "SGPG.L", "SMT.L", "SPX.L", "LAND.L", "EDV.L", "AUTO.L",
    "BRBY.L", "ENT.L", "SBRY.L", "KGF.L", "WTB.L", "MNDI.L", "SMIN.L", "BDEV.L",
    "TW.L", "JD.L", "HIK.L", "DARK.L", "FRAS.L", "HSX.L", "ITRK.L", "CRDA.L",
    "BME.L", "SMDS.L", "PSN.L", "RTO.L", "RS1.L", "HLN.L", "FLTR.L", "IAG.L",
    "EZJ.L", "WIZZ.L", "PHNX.L", "SJP.L", "JMAT.L", "DPLM.L", "HWDN.L",
    "FCIT.L", "BBOX.L", "BNKR.L", "CTY.L", "CGT.L", "TRIG.L", "UKW.L",
]

LSE_FTSE250_TOP = [
    "CINE.L", "FOUR.L", "GAW.L", "GBPG.L", "GNS.L", "HTWS.L", "IGG.L",
    "ITV.L", "JET2.L", "LTHM.L", "MGGT.L", "NXT.L", "OCDO.L", "PDG.L",
    "PETS.L", "RWS.L", "SFR.L", "TRN.L", "VCT.L", "VSVS.L",
    "WHR.L", "WIX.L", "YOU.L", "ASC.L", "BNZL.L", "DOCS.L",
    "FDEV.L", "FUTR.L", "GRG.L", "IPO.L",
]

UNIVERSES: dict[str, list[str]] = {
    "US_MEGA50": US_MEGA50,
    "US_LARGE100": US_LARGE100,
    "US_SP500": US_SP500_EXTENDED,
    "LSE_FTSE100": LSE_FTSE100,
    "LSE_FTSE250": LSE_FTSE250_TOP,
    "LSE_ALL": LSE_FTSE100 + LSE_FTSE250_TOP,
    "ALL": US_LARGE100 + LSE_FTSE100,
}

# ── Metrics ──────────────────────────────────────────────────────────────────

# Pass-1 metrics: available from yf.Ticker.info (no history needed)
INFO_METRICS: dict[str, str] = {
    "pe_ratio":       "trailingPE",
    "forward_pe":     "forwardPE",
    "peg_ratio":      "pegRatio",
    "market_cap":     "marketCap",
    "revenue_growth": "revenueGrowth",
    "profit_margin":  "profitMargins",
    "debt_to_equity": "debtToEquity",
    "beta":           "beta",
    "dividend_yield": "dividendYield",
    "free_cash_flow": "freeCashflow",
}

# Pass-2 metrics: require price history to compute
COMPUTED_METRICS = {"rsi_14d", "price_vs_200dma", "price_vs_50dma", "52w_pct"}

ALL_METRICS = set(INFO_METRICS.keys()) | COMPUTED_METRICS


def _compute_rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── Data fetching ────────────────────────────────────────────────────────────

def _fetch_info(symbol: str) -> dict[str, Any]:
    """Fetch .info dict for a single ticker. Returns empty dict on failure."""
    try:
        return yf.Ticker(symbol).info or {}
    except Exception as e:
        log.warning("Scanner: failed to fetch info for %s: %s", symbol, e)
        return {}


def _fetch_info_batch(symbols: list[str]) -> dict[str, dict]:
    """Fetch .info for a batch of tickers."""
    result = {}
    if not symbols:
        return result
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                result[sym] = tickers.tickers[sym].info or {}
            except Exception:
                result[sym] = {}
    except Exception as e:
        log.warning("Scanner: batch info fetch failed: %s", e)
        for sym in symbols:
            result[sym] = {}
    return result


def _extract_info_metrics(info: dict) -> dict[str, float | None]:
    """Extract pass-1 metric values from a yfinance info dict."""
    data: dict[str, float | None] = {}
    for metric_key, yf_field in INFO_METRICS.items():
        val = info.get(yf_field)
        data[metric_key] = float(val) if val is not None else None
    # Also grab display fields
    data["_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    data["_name"] = info.get("shortName") or info.get("longName") or ""
    data["_market_cap_raw"] = info.get("marketCap")
    data["_52w_high"] = info.get("fiftyTwoWeekHigh")
    data["_52w_low"] = info.get("fiftyTwoWeekLow")
    data["_50dma"] = info.get("fiftyDayAverage")
    data["_200dma"] = info.get("twoHundredDayAverage")
    return data


def _compute_history_metrics(symbol: str) -> dict[str, float | None]:
    """Fetch price history and compute RSI, DMA%, 52w% for a single ticker."""
    result: dict[str, float | None] = {
        "rsi_14d": None, "price_vs_200dma": None,
        "price_vs_50dma": None, "52w_pct": None,
    }
    try:
        hist = yf.Ticker(symbol).history(period="1y")
        if hist.empty or len(hist) < 15:
            return result
        closes = hist["Close"].tolist()
        price = closes[-1]

        result["rsi_14d"] = _compute_rsi(closes)

        if len(closes) >= 200:
            dma200 = float(np.mean(closes[-200:]))
            result["price_vs_200dma"] = ((price - dma200) / dma200) * 100
        if len(closes) >= 50:
            dma50 = float(np.mean(closes[-50:]))
            result["price_vs_50dma"] = ((price - dma50) / dma50) * 100

        high_52w = max(closes)
        if high_52w > 0:
            result["52w_pct"] = ((price - high_52w) / high_52w) * 100

    except Exception as e:
        log.warning("Scanner: history fetch failed for %s: %s", symbol, e)
    return result


# ── Condition evaluation ─────────────────────────────────────────────────────

def _evaluate_condition(value: float | None, condition: str, threshold: str) -> bool:
    """Check if a single metric value passes a condition."""
    if value is None:
        return False
    condition = condition.strip().lower()
    if condition == "between":
        parts = [p.strip() for p in threshold.split(",")]
        if len(parts) != 2:
            return False
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            return False
        return lo <= value <= hi
    try:
        thresh = float(threshold)
    except ValueError:
        return False
    if condition in (">", "above"):
        return value > thresh
    if condition in (">=",):
        return value >= thresh
    if condition in ("<", "below"):
        return value < thresh
    if condition in ("<=",):
        return value <= thresh
    if condition in ("=", "=="):
        return value == thresh
    return False


def _passes_all(stock_data: dict, conditions: list[dict]) -> bool:
    """Return True if stock_data passes ALL conditions."""
    for cond in conditions:
        metric = cond["metric"]
        value = stock_data.get(metric)
        if not _evaluate_condition(value, cond["condition"], cond["value"]):
            return False
    return True


# ── Scoring ──────────────────────────────────────────────────────────────────

def _compute_match_score(stock_data: dict, conditions: list[dict]) -> int:
    """
    Simple quality score 0-100.
    Each condition contributes points based on how much margin the stock
    passes by. More conditions passed with wider margins = higher score.
    """
    if not conditions:
        return 50
    total = 0
    for cond in conditions:
        metric = cond["metric"]
        value = stock_data.get(metric)
        if value is None:
            continue
        c = cond["condition"].strip().lower()
        try:
            if c == "between":
                parts = [float(p.strip()) for p in cond["value"].split(",")]
                lo, hi = parts[0], parts[1]
                mid = (lo + hi) / 2
                span = (hi - lo) / 2 if hi != lo else 1
                closeness = 1 - min(abs(value - mid) / span, 1)
                total += closeness * 100
            else:
                thresh = float(cond["value"])
                if thresh == 0:
                    total += 50
                elif c in (">", ">=", "above"):
                    ratio = value / thresh
                    total += min(ratio * 50, 100)
                elif c in ("<", "<=", "below"):
                    ratio = thresh / value if value != 0 else 1
                    total += min(ratio * 50, 100)
        except (ValueError, ZeroDivisionError):
            total += 50

    return int(round(total / len(conditions)))


# ── Format helpers ───────────────────────────────────────────────────────────

def _fmt_market_cap(mc: float | None) -> str:
    if mc is None:
        return ""
    if mc >= 1e12:
        return f"${mc/1e12:.1f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.1f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.0f}M"
    return f"${mc:,.0f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return ""
    return f"{val*100:.1f}%" if abs(val) < 1 else f"{val:.1f}%"


def _fmt_num(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}"


# ── Main scan ────────────────────────────────────────────────────────────────

def resolve_universe(universe_str: str) -> list[str]:
    """
    Resolve a universe string to a list of tickers.
    Supports: 'US_MEGA50', 'LSE_FTSE100', 'CUSTOM:AAPL,MSFT,...', or
    comma-separated universe names like 'US_MEGA50,LSE_FTSE100'.
    """
    universe_str = universe_str.strip()
    if not universe_str:
        return US_MEGA50

    # Custom ticker list
    if universe_str.upper().startswith("CUSTOM:"):
        tickers = [t.strip() for t in universe_str[7:].split(",") if t.strip()]
        return tickers

    # Single or comma-separated universe names
    combined = []
    for name in universe_str.split(","):
        name = name.strip().upper()
        if name in UNIVERSES:
            combined.extend(UNIVERSES[name])
        else:
            log.warning("Scanner: unknown universe '%s', skipping.", name)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for t in combined:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped if deduped else US_MEGA50


def run_scan(conditions: list[dict], universe: list[str]) -> list[dict]:
    """
    Run the stock scanner.
    conditions: list of {metric, condition, value}
    universe: list of yfinance ticker strings

    Returns list of matching stocks with all metric data, sorted by score.
    """
    if not conditions or not universe:
        return []

    # Split conditions into pass-1 (info) and pass-2 (history)
    pass1_conds = [c for c in conditions if c["metric"] not in COMPUTED_METRICS]
    pass2_conds = [c for c in conditions if c["metric"] in COMPUTED_METRICS]
    needs_pass2 = len(pass2_conds) > 0

    log.info("Scanner: %d tickers, %d conditions (pass1=%d, pass2=%d)",
             len(universe), len(conditions), len(pass1_conds), len(pass2_conds))

    # ── Pass 1: filter on .info metrics ──────────────────────────────────────
    survivors = []
    batch_size = 20
    for i in range(0, len(universe), batch_size):
        batch = universe[i:i + batch_size]
        if i > 0:
            time.sleep(0.5)  # gentle rate limiting
        infos = _fetch_info_batch(batch)
        for sym in batch:
            info = infos.get(sym, {})
            if not info:
                continue
            data = _extract_info_metrics(info)
            data["_symbol"] = sym
            if not pass1_conds or _passes_all(data, pass1_conds):
                survivors.append(data)
        log.info("  Batch %d-%d: %d/%d survived pass 1.",
                 i + 1, min(i + batch_size, len(universe)),
                 len([s for s in survivors[len(survivors) - len(batch):]]),
                 len(batch))

    log.info("Scanner pass 1 complete: %d/%d survived.", len(survivors), len(universe))

    # ── Pass 2: compute history metrics for survivors ────────────────────────
    if needs_pass2 and survivors:
        log.info("Scanner pass 2: computing history metrics for %d stocks...", len(survivors))
        final = []
        for stock in survivors:
            sym = stock["_symbol"]
            hist_data = _compute_history_metrics(sym)
            stock.update(hist_data)
            if _passes_all(stock, pass2_conds):
                final.append(stock)
            time.sleep(0.3)
        log.info("Scanner pass 2 complete: %d/%d survived.", len(final), len(survivors))
        survivors = final

    # ── Score and sort ───────────────────────────────────────────────────────
    results = []
    for stock in survivors:
        price = stock.get("_price")
        if not price:
            continue
        score = _compute_match_score(stock, conditions)
        results.append({
            "symbol": stock["_symbol"],
            "name": stock.get("_name", ""),
            "price": f"{price:.2f}",
            "pe": _fmt_num(stock.get("pe_ratio")),
            "fwd_pe": _fmt_num(stock.get("forward_pe")),
            "mkt_cap": _fmt_market_cap(stock.get("_market_cap_raw")),
            "rev_growth": _fmt_pct(stock.get("revenue_growth")),
            "margin": _fmt_pct(stock.get("profit_margin")),
            "de": _fmt_num(stock.get("debt_to_equity")),
            "rsi": _fmt_num(stock.get("rsi_14d")),
            "vs_200dma": _fmt_num(stock.get("price_vs_200dma")),
            "vs_52w": _fmt_num(stock.get("52w_pct")),
            "beta": _fmt_num(stock.get("beta"), 2),
            "div_yield": _fmt_pct(stock.get("dividend_yield")),
            "score": score,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    log.info("Scanner: %d matches found.", len(results))
    return results
