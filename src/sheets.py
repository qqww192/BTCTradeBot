"""
sheets.py
Google Sheets integration for the T212 Portfolio Checker.

Manages a single spreadsheet with three tabs:
  - Portfolio:       auto-populated from T212, with configurable priority
  - Watchlist:       user-added symbols for research
  - Market Overview: daily macro/sector analysis

Authentication: Google Service Account via GOOGLE_SA_JSON env var.
Sheet ID: GOOGLE_SHEET_ID env var (must be set to an existing spreadsheet).
The spreadsheet must be shared with the service account email as Editor.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

SHEET_ID: str = os.environ.get("GOOGLE_SHEET_ID", "")

# Tab names
TAB_PORTFOLIO = "Portfolio"
TAB_WATCHLIST = "Watchlist"
TAB_MARKET = "Market Overview"

# Column headers for each tab
PORTFOLIO_HEADERS = [
    "Pie", "Symbol", "Amount", "Price", "Weight %", "Analysis", "Last Updated",
]

WATCHLIST_HEADERS = [
    "Symbol", "Analysis", "Last Updated",
]

MARKET_HEADERS = [
    "Date", "Indicator", "Value", "Change %", "Sentiment", "Analysis", "Last Updated",
]


def _get_credentials() -> service_account.Credentials:
    sa_json = os.environ.get("GOOGLE_SA_JSON", "")
    if not sa_json:
        raise EnvironmentError(
            "GOOGLE_SA_JSON environment variable is not set. "
            "See docs/setup.md for configuration."
        )
    sa_dict = json.loads(sa_json)
    return service_account.Credentials.from_service_account_info(sa_dict, scopes=SCOPES)


def _ensure_tabs_exist(sheets_service, sheet_id: str) -> None:
    """Ensure the three required tabs exist; create any that are missing."""
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    existing_tabs = {s["properties"]["title"] for s in spreadsheet["sheets"]}

    required_tabs = [TAB_PORTFOLIO, TAB_WATCHLIST, TAB_MARKET]
    missing = [t for t in required_tabs if t not in existing_tabs]
    if not missing:
        return

    requests = [{"addSheet": {"properties": {"title": tab}}} for tab in missing]
    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id, body={"requests": requests},
    ).execute()

    # Write headers for newly created tabs
    header_data = []
    headers_map = {
        TAB_PORTFOLIO: PORTFOLIO_HEADERS,
        TAB_WATCHLIST: WATCHLIST_HEADERS,
        TAB_MARKET: MARKET_HEADERS,
    }
    for tab in missing:
        header_data.append({"range": f"'{tab}'!A1", "values": [headers_map[tab]]})
    if header_data:
        sheets_service.spreadsheets().values().batchUpdate(
            spreadsheetId=sheet_id,
            body={"valueInputOption": "RAW", "data": header_data},
        ).execute()

    log.info("Created missing tabs: %s", ", ".join(missing))


class SheetManager:
    """High-level interface to read/write the T212 Portfolio Tracker sheet."""

    def __init__(self):
        if not SHEET_ID:
            raise EnvironmentError(
                "GOOGLE_SHEET_ID environment variable is not set. "
                "Set it to the ID of an existing Google Sheet shared with the service account."
            )
        creds = _get_credentials()
        self.sheets = build("sheets", "v4", credentials=creds)
        self.sheet_id = SHEET_ID
        _ensure_tabs_exist(self.sheets, self.sheet_id)
        self.url = f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/edit"

    # ── Portfolio tab ──────────────────────────────────────────────────────────

    def sync_portfolio(self, positions: list[dict[str, Any]], prices: dict[str, float] | None = None) -> None:
        """
        Sync T212 positions into the Portfolio tab.
        Merges with existing rows to preserve analysis.
        prices: optional dict of symbol -> live price from yfinance.
        """
        existing = self._read_tab(TAB_PORTFOLIO)
        existing_by_symbol: dict[str, list] = {}
        for row in existing:
            if len(row) >= 2 and row[1]:  # Symbol in column B
                existing_by_symbol[row[1]] = row

        prices = prices or {}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Calculate total portfolio value for weight
        total_value = 0
        pos_data = []
        for pos in positions:
            ticker = pos.get("ticker", "N/A")
            qty = float(pos.get("quantity", 0))
            price = prices.get(ticker) or float(pos.get("currentPrice", 0))
            value = qty * price
            total_value += value
            pos_data.append((pos, ticker, qty, price, value))

        rows = []
        for pos, ticker, qty, price, value in pos_data:
            pie_name = pos.get("pieAccountName", "")
            weight = f"{value / total_value * 100:.1f}" if total_value else "0"

            old = existing_by_symbol.get(ticker, [])
            analysis = old[5] if len(old) > 5 else ""

            rows.append([pie_name, ticker, qty, price, weight, analysis, now])

        # Sort by weight descending
        rows.sort(key=lambda r: float(r[4]) if r[4] else 0, reverse=True)

        all_rows = [PORTFOLIO_HEADERS] + rows
        self._write_tab(TAB_PORTFOLIO, all_rows)
        log.info("Portfolio tab synced with %d positions.", len(rows))

    def get_portfolio_for_analysis(self) -> list[dict]:
        """
        Returns portfolio stocks sorted by weight (highest first).
        Columns: Pie, Symbol, Amount, Price, Weight %, Analysis, Last Updated
        """
        rows = self._read_tab(TAB_PORTFOLIO)
        stocks = []
        for row in rows:
            if len(row) < 2 or not row[1]:
                continue
            stocks.append({
                "symbol": row[1],
                "pie": row[0] if len(row) > 0 else "",
                "amount": row[2] if len(row) > 2 else 0,
                "price": row[3] if len(row) > 3 else 0,
                "weight": row[4] if len(row) > 4 else "0",
                "last_updated": row[6] if len(row) > 6 else "",
            })
        # Sort by weight descending (analyse biggest positions first)
        stocks.sort(key=lambda s: float(s.get("weight", 0)), reverse=True)
        return stocks

    def update_portfolio_analysis(self, symbol: str, analysis: str) -> None:
        """Update the analysis result and timestamp for a portfolio stock."""
        rows = self._read_tab(TAB_PORTFOLIO)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for i, row in enumerate(rows):
            if len(row) >= 2 and row[1] == symbol:
                while len(row) < 7:
                    row.append("")
                row[5] = analysis       # Analysis
                row[6] = now            # Last Updated
                self.sheets.spreadsheets().values().update(
                    spreadsheetId=self.sheet_id,
                    range=f"'{TAB_PORTFOLIO}'!A{i + 1}",
                    valueInputOption="RAW",
                    body={"values": [row]},
                ).execute()
                return

    # ── Watchlist tab ──────────────────────────────────────────────────────────

    def get_watchlist(self) -> list[dict]:
        """Returns watchlist symbols that have been filled in by the user."""
        rows = self._read_tab(TAB_WATCHLIST)
        watchlist = []
        for i, row in enumerate(rows):
            if len(row) >= 1 and row[0]:
                watchlist.append({
                    "symbol": row[0],
                    "existing_analysis": row[1] if len(row) > 1 else "",
                    "row_index": i + 1,
                })
        return watchlist

    def update_watchlist_analysis(self, symbol: str, analysis: str) -> None:
        """Update analysis for a watchlist symbol."""
        rows = self._read_tab(TAB_WATCHLIST)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for i, row in enumerate(rows):
            if len(row) >= 1 and row[0] == symbol:
                while len(row) < 3:
                    row.append("")
                row[1] = analysis    # Analysis
                row[2] = now         # Last Updated
                self.sheets.spreadsheets().values().update(
                    spreadsheetId=self.sheet_id,
                    range=f"'{TAB_WATCHLIST}'!A{i + 1}",
                    valueInputOption="RAW",
                    body={"values": [row]},
                ).execute()
                return

    # ── Market Overview tab ────────────────────────────────────────────────────

    def write_market_overview(self, entries: list[dict]) -> None:
        """
        Write daily market overview entries.
        Each entry: {indicator, value, change_pct, sentiment, analysis}
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        existing = self._read_tab(TAB_MARKET)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Remove today's existing entries (will be replaced)
        kept = [r for r in existing if not (len(r) >= 1 and r[0] == today)]

        new_rows = []
        for entry in entries:
            new_rows.append([
                today,
                entry.get("indicator", ""),
                entry.get("value", ""),
                entry.get("change_pct", ""),
                entry.get("sentiment", ""),
                entry.get("analysis", ""),
                now,
            ])

        all_rows = [MARKET_HEADERS] + kept + new_rows
        self._write_tab(TAB_MARKET, all_rows)
        log.info("Market Overview updated with %d entries.", len(new_rows))

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _read_tab(self, tab_name: str) -> list[list]:
        """Read all rows from a tab (excluding header)."""
        try:
            result = self.sheets.spreadsheets().values().get(
                spreadsheetId=self.sheet_id,
                range=f"'{tab_name}'!A:Z",
            ).execute()
            rows = result.get("values", [])
            return rows[1:] if len(rows) > 1 else []  # skip header
        except HttpError as e:
            log.warning("Failed to read tab '%s': %s", tab_name, e)
            return []

    def _write_tab(self, tab_name: str, rows: list[list]) -> None:
        """Overwrite an entire tab with the given rows (including header)."""
        # Clear existing content
        self.sheets.spreadsheets().values().clear(
            spreadsheetId=self.sheet_id,
            range=f"'{tab_name}'!A:Z",
        ).execute()
        # Write new content
        if rows:
            self.sheets.spreadsheets().values().update(
                spreadsheetId=self.sheet_id,
                range=f"'{tab_name}'!A1",
                valueInputOption="RAW",
                body={"values": rows},
            ).execute()


def _detect_market(ticker: str) -> str:
    """Simple heuristic to detect US vs UK market from ticker format."""
    # T212 UK tickers often end with _EQ (e.g., BARC_EQ), or have suffixes like .L
    if ticker.endswith("_EQ") or ".L" in ticker or ticker.endswith("_LSE"):
        return "UK"
    # Most other tickers are US
    return "US"
