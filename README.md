# BTC/USDT Grid Trading Bot

Adaptive spot grid trader for crypto.com Exchange running on an Oracle Cloud Always Free ARM VM.

## How it works

```
Oracle Cloud VM (cron scheduler)
        │
        ├─► every 5 min  → src/trading/grid_trader.py
        │                     fill detection → risk check → order placement
        │
        ├─► every 4 hrs  → src/trading/regime_classifier.py
        │                     ATR-14 + Bollinger Band Width → data/regime.json
        │
        ├─► daily 08:00  → src/trading/daily_reporter.py
        │                     P&L summary → Telegram
        │
        └─► Sunday 23:00 → src/trading/gemini_optimizer.py
                              Gemini AI review → config/grid_params.json
```

## Setup

See **[oracle_deployment.md](oracle_deployment.md)** for the full step-by-step VM provisioning and deployment guide.

**Secrets required in `.env` on the Oracle VM:**

| Variable | Description |
|---|---|
| `CDX_API_KEY` | crypto.com Exchange API key (Trade permission only) |
| `CDX_API_SECRET` | crypto.com Exchange API secret |
| `GEMINI_API_KEY` | Google AI Studio → Get API Key |
| `TELEGRAM_BOT_TOKEN` | Telegram → @BotFather → /newbot |
| `TELEGRAM_CHAT_ID` | Telegram → @userinfobot |
| `TOTAL_CAPITAL_GBP` | Total capital deployed (default: 150) |
| `GBP_USD_RATE` | GBP/USD rate for P&L conversion (default: 1.27) |
| `KILL_SWITCH_PCT` | Weekly loss limit as fraction of capital (default: 0.10) |

## Key constraints

- All orders are `POST_ONLY` limit orders — no market orders, ever
- Maximum `capital_pct`: 0.80 (keep ≥20% as reserve)
- Weekly kill switch at −10% of total capital; auto-resets Monday 00:00 UTC
- Spot only — no leverage, no margin, no derivatives
- API key has Trade permission but NOT withdrawal permission
