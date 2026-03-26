"""
analyse.py
Gemini analysis engine with request budget management.

Uses yfinance data for real financial context.
Produces concise, actionable analysis.
"""

import os
import re
import logging
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

log = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
MAX_DAILY_REQUESTS = 19


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "See docs/setup.md for configuration."
        )
    return genai.Client(api_key=api_key)


def _call_gemini(client: genai.Client, prompt: str) -> str:
    """Call Gemini with retry on 429. Returns the response text."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            break
        except genai_errors.ClientError as exc:
            if exc.code == 429 and attempt < max_retries:
                wait = 60 * attempt
                match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
                if match:
                    wait = float(match.group(1)) + 2
                log.warning(
                    "Gemini rate-limited (attempt %d/%d). Waiting %.0fs.",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise

    text = response.text
    if not text:
        log.error("Gemini returned no text. Full response: %s", response)
        raise RuntimeError("Gemini returned an empty response.")
    return text


# ── Watchlist Analysis ─────────────────────────────────────────────────────


def analyse_watchlist_symbol(client: genai.Client, symbol: str, financial_context: str) -> str:
    """Concise analysis of a watchlist symbol."""
    prompt = f"""You are a stock analyst. Give a BRIEF analysis of {symbol}.

Live data: {financial_context}

In 3-4 sentences max, cover:
1. What the company does and its moat
2. Is it undervalued or overvalued? (use the P/E, growth, margins above)
3. Key risk and key catalyst
4. Verdict: BUY / HOLD / SELL with one-line reasoning

Be direct and specific with numbers. No headers or bullet points."""
    return _call_gemini(client, prompt)


# ── Market Overview (with VIX) ────────────────────────────────────────────


def analyse_market_overview(
    client: genai.Client,
    market_data: dict[str, dict],
    vix_data: dict,
    positions: list[dict[str, Any]],
) -> str:
    """
    Concise market overview using real index data + VIX.
    Returns a short analysis string.
    """
    # Build market data summary
    index_lines = []
    for name, data in market_data.items():
        change = data.get("change_pct", 0)
        direction = "+" if change >= 0 else ""
        index_lines.append(f"- {name}: {data.get('price', 0):,.0f} ({direction}{change:.2f}%)")

    index_summary = "\n".join(index_lines)

    vix_current = vix_data.get("current", 0)
    vix_change = vix_data.get("change_pct", 0)
    vix_sentiment = vix_data.get("sentiment", "N/A")

    ticker_list = ", ".join(pos.get("ticker", "N/A") for pos in positions[:30])

    prompt = f"""You are a macro strategist. Give a BRIEF daily market briefing (max 5 sentences).

Market data right now:
{index_summary}
- VIX: {vix_current:.2f} ({'+' if vix_change >= 0 else ''}{vix_change:.1f}%) — {vix_sentiment}

My holdings: {ticker_list}

Cover: overall market direction, VIX signal, which of my holdings benefit/face headwinds, one key event to watch.
Be direct. No headers or bullet points. Use numbers."""
    return _call_gemini(client, prompt)


# ── Stock Analysis (concise) ──────────────────────────────────────────────


def analyse_stock_advanced(
    client: genai.Client,
    symbol: str,
    financial_context: str,
    amount: Any,
    price: Any,
    weight: Any,
) -> str:
    """
    Concise stock analysis using real yfinance data.
    Inspired by: undervalue screener, sentiment vs reality gap, risk-adjusted analysis.
    """
    prompt = f"""You are a senior equity analyst. Analyse {symbol} BRIEFLY (max 4-5 sentences).

Live financial data: {financial_context}
My position: {amount} shares at ${price}, portfolio weight: {weight}%

Cover in your response:
1. Is it undervalued? (P/E vs industry, growth rate, margins)
2. Sentiment vs fundamentals — is the market mispricing it?
3. Top risk and nearest catalyst
4. Action: STRONG BUY / BUY / HOLD / SELL / STRONG SELL with target price

Be specific with numbers. No headers, no bullet points. Just a tight analytical paragraph."""
    return _call_gemini(client, prompt)


# ── Budget-aware orchestrator ──────────────────────────────────────────────


class AnalysisBudget:
    """Tracks Gemini API request budget for a single run."""

    def __init__(self, max_requests: int = MAX_DAILY_REQUESTS):
        self.max_requests = max_requests
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.max_requests - self.used

    def consume(self) -> bool:
        """Consume one request. Returns True if budget allows, False if exhausted."""
        if self.used >= self.max_requests:
            return False
        self.used += 1
        log.info("Gemini request %d/%d used.", self.used, self.max_requests)
        return True

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_requests
