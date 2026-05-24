"""
Betfair Exchange API client — thin wrapper around betfairlightweight.

Handles login, market discovery, price fetching, and order placement.
All credentials are read from environment variables; never hardcoded.

Betfair API concepts
--------------------
- Event type  : top-level category (Politics, Horse Racing, etc.)
- Market      : a specific question within an event ("UK Election winner")
- Runner      : one outcome within a market ("Labour", "Conservative")
- Back bet    : betting that a runner WINS (we always back, never lay)
- Odds        : decimal format — 1.10 means 91% implied probability
- Commission  : 2–5% of net winnings per market (not per trade)
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import betfairlightweight
from betfairlightweight import filters as bf

# Event type IDs we scan for near-certainty opportunities.
# Resolved dynamically via list_event_types() on first run if needed.
POLITICS_EVENT_TYPE_ID  = "2378961"
SPECIALS_EVENT_TYPE_ID  = "10"       # Betfair "Specials" market group
CURRENT_AFFAIRS_TYPE_ID = "26420387"

TARGET_EVENT_TYPE_IDS = [
    POLITICS_EVENT_TYPE_ID,
    SPECIALS_EVENT_TYPE_ID,
    CURRENT_AFFAIRS_TYPE_ID,
]

BETFAIR_TIMEOUT = 10  # seconds


class BetfairError(Exception):
    """Raised when a Betfair API call fails or returns an error status."""
    pass


class BetfairClient:
    def __init__(self):
        self._trading = betfairlightweight.APIClient(
            username=os.environ["BETFAIR_USERNAME"],
            password=os.environ["BETFAIR_PASSWORD"],
            app_key=os.environ["BETFAIR_APP_KEY"],
        )
        self._logged_in = False

    # ------------------------------------------------------------------ #
    #  Session                                                             #
    # ------------------------------------------------------------------ #

    def login(self) -> None:
        self._trading.login()
        self._logged_in = True

    def logout(self) -> None:
        if self._logged_in:
            try:
                self._trading.logout()
            except Exception:
                pass
            self._logged_in = False

    # ------------------------------------------------------------------ #
    #  Market discovery                                                    #
    # ------------------------------------------------------------------ #

    def find_near_certainty_markets(
        self,
        max_odds: float,
        max_hours: int,
    ) -> list[dict]:
        """
        Return a list of candidate bets: markets where at least one runner
        has best available back odds <= max_odds, closing within max_hours.

        Each item in the returned list is:
        {
            "market_id":    str,
            "market_name":  str,
            "close_time":   datetime (UTC),
            "runner_id":    int,
            "runner_name":  str,
            "best_back":    float,   # best available back price
            "available":    float,   # £ available at that price
        }
        """
        now    = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=max_hours)

        market_filter = bf.market_filter(
            event_type_ids=TARGET_EVENT_TYPE_IDS,
            market_start_time=bf.time_range(
                to=cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            in_play_only=False,
            market_countries=["GB", "US", "AU", "IE"],
        )

        try:
            catalogues = self._trading.betting.list_market_catalogue(
                filter=market_filter,
                market_projection=[
                    "MARKET_START_TIME",
                    "RUNNER_DESCRIPTION",
                    "EVENT_TYPE",
                    "EVENT",
                ],
                sort="FIRST_TO_START",
                max_results=200,
            )
        except Exception as exc:
            raise BetfairError(f"list_market_catalogue failed: {exc}") from exc

        if not catalogues:
            return []

        market_ids = [m.market_id for m in catalogues]

        try:
            books = self._trading.betting.list_market_book(
                market_ids=market_ids,
                price_projection=bf.price_projection(
                    price_data=["EX_BEST_OFFERS"],
                    ex_best_offers_overrides=bf.ex_best_offers_overrides(
                        best_prices_depth=1,
                    ),
                ),
            )
        except Exception as exc:
            raise BetfairError(f"list_market_book failed: {exc}") from exc

        # Index catalogues by market_id for runner name lookup
        cat_index = {m.market_id: m for m in catalogues}

        candidates = []
        for book in books:
            if book.status != "OPEN":
                continue

            cat = cat_index.get(book.market_id)
            if not cat:
                continue

            runner_names = {
                r.selection_id: r.runner_name
                for r in (cat.runners or [])
            }

            for runner in book.runners:
                if runner.status != "ACTIVE":
                    continue

                backs = runner.ex.available_to_back if runner.ex else []
                if not backs:
                    continue

                best_back = backs[0].price
                available = backs[0].size

                if best_back <= max_odds:
                    candidates.append({
                        "market_id":   book.market_id,
                        "market_name": cat.market_name,
                        "close_time":  cat.market_start_time,
                        "runner_id":   runner.selection_id,
                        "runner_name": runner_names.get(runner.selection_id, "Unknown"),
                        "best_back":   best_back,
                        "available":   available,
                    })

        # Sort by closest to resolution first
        candidates.sort(key=lambda x: x["close_time"])
        return candidates

    # ------------------------------------------------------------------ #
    #  Order placement                                                     #
    # ------------------------------------------------------------------ #

    def place_back_bet(
        self,
        market_id: str,
        runner_id: int,
        odds: float,
        stake_gbp: float,
        customer_ref: Optional[str] = None,
    ) -> dict:
        """
        Place a limit back bet. Uses LAPSE persistence so the order is
        cancelled automatically if unmatched when the market goes in-play.

        Returns a dict with bet_id, status, and placed_date on success.
        Raises BetfairError on failure.
        """
        instruction = bf.place_instruction(
            selection_id=runner_id,
            side="BACK",
            order_type="LIMIT",
            limit_order=bf.limit_order(
                size=round(stake_gbp, 2),
                price=odds,
                persistence_type="LAPSE",
            ),
        )

        try:
            result = self._trading.betting.place_orders(
                market_id=market_id,
                instructions=[instruction],
                customer_ref=customer_ref,
            )
        except Exception as exc:
            raise BetfairError(f"place_orders failed: {exc}") from exc

        if result.status != "SUCCESS":
            raise BetfairError(
                f"place_orders returned status={result.status} "
                f"errors={result.error_code}"
            )

        instr_report = result.instruction_reports[0]
        if instr_report.status != "SUCCESS":
            raise BetfairError(
                f"Instruction failed: {instr_report.error_code}"
            )

        return {
            "bet_id":      instr_report.bet_id,
            "status":      instr_report.status,
            "placed_date": instr_report.placed_date.isoformat()
            if instr_report.placed_date else None,
        }

    # ------------------------------------------------------------------ #
    #  Open / settled bets                                                 #
    # ------------------------------------------------------------------ #

    def get_open_bets(self) -> list[dict]:
        """Return currently open (unmatched + matched) bets."""
        try:
            orders = self._trading.betting.list_current_orders()
        except Exception as exc:
            raise BetfairError(f"list_current_orders failed: {exc}") from exc

        return [
            {
                "bet_id":       o.bet_id,
                "market_id":    o.market_id,
                "runner_id":    o.selection_id,
                "side":         o.side,
                "price":        o.price_size.price,
                "size_matched": o.size_matched,
                "size_remaining": o.size_remaining,
                "status":       o.status,
            }
            for o in (orders.current_orders or [])
        ]

    def get_settled_bets(self, from_dt: datetime) -> list[dict]:
        """Return bets settled after from_dt."""
        try:
            cleared = self._trading.betting.list_cleared_orders(
                bet_status="SETTLED",
                settled_date_range=bf.time_range(
                    from_=from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
        except Exception as exc:
            raise BetfairError(f"list_cleared_orders failed: {exc}") from exc

        return [
            {
                "bet_id":       o.bet_id,
                "market_id":    o.market_id,
                "runner_id":    o.selection_id,
                "side":         o.side,
                "price":        o.price_requested,
                "size_matched": o.size_settled,
                "profit":       o.profit,           # net after commission
                "settled_date": o.settled_date.isoformat()
                if o.settled_date else None,
            }
            for o in (cleared.cleared_orders or [])
        ]
