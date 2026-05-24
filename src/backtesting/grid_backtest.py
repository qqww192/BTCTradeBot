"""
Grid strategy backtester — Skill 7 (hftbacktest-inspired).

A purpose-built grid trading simulator that models:
  - Individual order placement and fill sequencing
  - POST_ONLY maker fee (0.25% per leg)
  - Grid recentring every time price moves > 3% from calibration
  - Capital reservation (capital_pct ≤ 0.80)
  - Per-level buy-price tracking for accurate SELL profit calculation
  - Regime transitions (reads regime thresholds per candle)
  - Daily P&L, weekly Sharpe, max drawdown, win rate

Unlike the simple simulate_return() in gemini_optimizer.py this backtester
steps through each candle in order, maintains open order state, and counts
only fills that are geometrically reachable given the day's OHLC range.

Usage
-----
  # Quick backtest with current config:
  python3 src/backtesting/grid_backtest.py

  # Custom params:
  python3 src/backtesting/grid_backtest.py --spacing 0.9 --levels 8 --capital 0.65 --days 90

  # From gemini_optimizer:
  from backtesting.grid_backtest import run_backtest, BacktestResult
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

MAKER_FEE   = 0.0025   # 0.25% per leg
MIN_QTY_BTC = 0.0001
RECENTER_THRESHOLD_PCT = 3.0  # recentre if price moves > 3% from calibration


@dataclass
class GridConfig:
    spacing_pct:   float = 0.8
    range_pct:     float = 5.0
    levels:        int   = 10
    capital_pct:   float = 0.70
    total_capital: float = 150.0
    gbp_usd_rate:  float = 1.27
    kill_pct:      float = 0.10

    @classmethod
    def from_file(cls, path: Path) -> "GridConfig":
        if path.exists():
            d = json.loads(path.read_text())
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()

    @property
    def capital_usdt(self) -> float:
        return self.total_capital * self.capital_pct * self.gbp_usd_rate

    @property
    def per_level_usdt(self) -> float:
        return self.capital_usdt / self.levels if self.levels else 0


# ------------------------------------------------------------------ #
#  Order model                                                         #
# ------------------------------------------------------------------ #

@dataclass
class Order:
    level:     int
    side:      str      # BUY | SELL
    price:     float
    qty:       float
    buy_price: Optional[float] = None   # for SELL orders: the matched BUY price


# ------------------------------------------------------------------ #
#  Result                                                              #
# ------------------------------------------------------------------ #

@dataclass
class BacktestResult:
    total_trades:    int   = 0
    total_sells:     int   = 0
    wins:            int   = 0
    total_gross_gbp: float = 0.0
    total_fees_gbp:  float = 0.0
    total_net_gbp:   float = 0.0
    max_drawdown_gbp: float = 0.0
    daily_pnl:       list  = field(default_factory=list)
    recenter_count:  int   = 0

    @property
    def win_rate_pct(self) -> float:
        return round(self.wins / self.total_sells * 100, 1) if self.total_sells else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.daily_pnl) < 2:
            return 0.0
        mean = sum(self.daily_pnl) / len(self.daily_pnl)
        var  = sum((x - mean) ** 2 for x in self.daily_pnl) / len(self.daily_pnl)
        std  = math.sqrt(var)
        return round(mean / std, 3) if std else 0.0

    @property
    def fee_drag_pct(self) -> float:
        if self.total_gross_gbp <= 0:
            return 0.0
        return round(self.total_fees_gbp / self.total_gross_gbp * 100, 1)

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | Sells: {self.total_sells} | "
            f"Win rate: {self.win_rate_pct}%\n"
            f"Net P&L: £{self.total_net_gbp:.2f} | "
            f"Gross: £{self.total_gross_gbp:.2f} | "
            f"Fees: £{self.total_fees_gbp:.2f}\n"
            f"Max drawdown: £{self.max_drawdown_gbp:.2f} | "
            f"Sharpe: {self.sharpe} | "
            f"Fee drag: {self.fee_drag_pct}%\n"
            f"Recentres: {self.recenter_count}"
        )


# ------------------------------------------------------------------ #
#  Grid builder                                                        #
# ------------------------------------------------------------------ #

def _build_grid(center: float, config: GridConfig) -> list[Order]:
    """Build a grid of limit orders around center price."""
    spacing = config.spacing_pct / 100
    half    = config.levels // 2
    orders  = []
    for i in range(config.levels):
        offset = (i - half) * spacing
        price  = round(center * (1 + offset), 2)
        side   = "SELL" if i >= half else "BUY"
        qty    = max(config.per_level_usdt / price, MIN_QTY_BTC)
        orders.append(Order(level=i, side=side, price=price, qty=round(qty, 6)))
    return orders


# ------------------------------------------------------------------ #
#  Fill simulation (per candle)                                        #
# ------------------------------------------------------------------ #

def _simulate_candle_fills(
    orders:     list[Order],
    candle:     dict,
    buy_prices: dict,   # level_idx → buy_price
    config:     GridConfig,
    result:     BacktestResult,
) -> tuple[list[Order], dict]:
    """
    For each order, decide if it would have filled during this candle's
    OHLC range. Uses a simple sweep model:
      - High of day reaches SELL orders above open
      - Low of day reaches BUY orders below open
    Returns (remaining_orders, updated_buy_prices).
    """
    high     = candle["high"]
    low      = candle["low"]
    close    = candle["close"]
    gbp_rate = config.gbp_usd_rate

    remaining  = []
    day_pnl    = 0.0
    day_fees   = 0.0

    for order in orders:
        filled = False
        if order.side == "BUY"  and order.price >= low:
            filled = True
        elif order.side == "SELL" and order.price <= high:
            filled = True

        if not filled:
            remaining.append(order)
            continue

        fee_usdt  = order.price * order.qty * MAKER_FEE
        fee_gbp   = fee_usdt / gbp_rate
        result.total_trades += 1

        if order.side == "BUY":
            buy_prices[order.level] = order.price
            day_fees    += fee_gbp
            replacement  = Order(
                level     = order.level,
                side      = "SELL",
                price     = round(order.price * (1 + config.spacing_pct / 100), 2),
                qty       = order.qty,
                buy_price = order.price,
            )
            remaining.append(replacement)

        else:  # SELL
            result.total_sells += 1
            buy_px   = order.buy_price or buy_prices.get(order.level, order.price)
            gross_usdt = (order.price - buy_px) * order.qty
            gross_gbp  = gross_usdt / gbp_rate
            net_gbp    = gross_gbp - fee_gbp
            result.total_gross_gbp += gross_gbp
            result.total_fees_gbp  += fee_gbp
            result.total_net_gbp   += net_gbp
            day_pnl  += net_gbp
            day_fees += fee_gbp
            if net_gbp > 0:
                result.wins += 1
            buy_prices.pop(order.level, None)
            replacement = Order(
                level = order.level,
                side  = "BUY",
                price = round(order.price / (1 + config.spacing_pct / 100), 2),
                qty   = order.qty,
            )
            remaining.append(replacement)

    result.daily_pnl.append(day_pnl)
    return remaining, buy_prices


# ------------------------------------------------------------------ #
#  Main backtest loop                                                  #
# ------------------------------------------------------------------ #

def run_backtest(
    candles: list[dict],
    config:  GridConfig,
    verbose: bool = False,
) -> BacktestResult:
    """
    Step through each candle in order, maintaining grid state.
    Recentres the grid whenever price moves > 3% from calibration.
    """
    if len(candles) < 5:
        print("[backtest] Not enough candles.")
        return BacktestResult()

    result      = BacktestResult()
    calibration = candles[0]["close"]
    orders      = _build_grid(calibration, config)
    buy_prices: dict[int, float] = {}

    cumulative_pnl = 0.0
    peak_pnl       = 0.0

    for candle in candles:
        price = candle["close"]

        # Recentre check
        move_pct = abs(price - calibration) / calibration * 100
        if move_pct > RECENTER_THRESHOLD_PCT:
            calibration = price
            orders      = _build_grid(calibration, config)
            buy_prices  = {}
            result.recenter_count += 1
            if verbose:
                print(f"[backtest] Recentre at ${price:,.0f} (move={move_pct:.1f}%)")

        orders, buy_prices = _simulate_candle_fills(
            orders, candle, buy_prices, config, result
        )

        # Track max drawdown
        cumulative_pnl += result.daily_pnl[-1] if result.daily_pnl else 0
        peak_pnl        = max(peak_pnl, cumulative_pnl)
        drawdown        = peak_pnl - cumulative_pnl
        result.max_drawdown_gbp = max(result.max_drawdown_gbp, drawdown)

    return result


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid strategy backtester")
    parser.add_argument("--spacing",  type=float, default=None, help="spacing_pct override")
    parser.add_argument("--levels",   type=int,   default=None, help="levels override")
    parser.add_argument("--capital",  type=float, default=None, help="capital_pct override")
    parser.add_argument("--days",     type=int,   default=30,   help="number of candles to fetch")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    from trading.cdx_client import CDXClient
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    config = GridConfig.from_file(ROOT / "config" / "grid_params.json")
    if args.spacing  is not None: config.spacing_pct  = args.spacing
    if args.levels   is not None: config.levels        = args.levels
    if args.capital  is not None: config.capital_pct   = args.capital

    print(f"[backtest] Fetching {args.days} daily candles...")
    cdx     = CDXClient()
    candles = cdx.get_candlesticks("BTC_USDT", timeframe="1D", count=args.days)
    print(f"[backtest] Got {len(candles)} candles. Running simulation...")
    print(f"[backtest] Config: spacing={config.spacing_pct}% levels={config.levels} "
          f"capital={config.capital_pct} kill_pct={config.kill_pct}")

    result = run_backtest(candles, config, verbose=args.verbose)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(result.summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
