"""
analyse.py
Sends the portfolio data to Google Gemini for analysis and returns a markdown report.

Uses the google-genai SDK with Search grounding for up-to-date market context.
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


def _get_client() -> genai.Client:
    """Create and return a Gemini client configured with the API key."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "See docs/setup.md for how to obtain and configure it."
        )
    return genai.Client(api_key=api_key)


def _build_prompt(positions: list[dict[str, Any]], run_date: str) -> str:
    """Builds the analysis prompt from portfolio positions."""
    lines = [f"Portfolio snapshot as of {run_date}\n"]
    lines.append(f"{'Ticker':<12} {'Qty':>8} {'Avg Price':>10} {'Current':>10} {'P/L':>10}")
    lines.append("─" * 54)

    for pos in positions:
        ticker = pos.get("ticker", "N/A")
        qty = pos.get("quantity", 0)
        avg_price = pos.get("averagePrice", 0)
        current = pos.get("currentPrice", 0)
        ppl = pos.get("ppl", 0)
        lines.append(f"{ticker:<12} {qty:>8.2f} {avg_price:>10.2f} {current:>10.2f} {ppl:>10.2f}")

    portfolio_table = "\n".join(lines)

    prompt = f"""You are a financial analyst assistant. Analyse the following Trading 212 portfolio
and produce a concise markdown report.

{portfolio_table}

For each position, provide:
1. A brief assessment of the holding (1–2 sentences)
2. Key recent news or events affecting the stock
3. A risk rating from 1 (low) to 10 (high)

End with an overall portfolio summary including:
- Total diversification assessment
- Top risks across the portfolio
- Suggested actions (if any)

Format the output as clean markdown with headers and tables where appropriate.
"""
    return prompt


def analyse_portfolio(positions: list[dict[str, Any]], run_date: str) -> str:
    """
    Sends portfolio data to Gemini for analysis.
    Returns the analysis as a markdown string.
    """
    client = _get_client()

    prompt = _build_prompt(positions, run_date)

    max_retries = 3
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            break
        except genai_errors.ClientError as exc:
            if exc.code == 429 and attempt < max_retries:
                # Parse retry delay from error message, fall back to exponential backoff
                wait = 60 * attempt  # default backoff
                match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
                if match:
                    wait = float(match.group(1)) + 2  # add small buffer
                log.warning(
                    "Gemini rate-limited (attempt %d/%d). Waiting %.0fs before retry.",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise

    report = response.text
    if not report:
        # Log full response for debugging, then raise a clear error
        log.error("Gemini returned no text. Full response: %s", response)
        raise RuntimeError(
            "Gemini returned an empty response. This can happen when the "
            "response is blocked by safety filters or the model produces no "
            "text output. Check the logs above for the full API response."
        )
    log.info(f"Gemini analysis complete — {len(report)} chars.")
    return report
