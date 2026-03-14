# GOAT Live Trading Bot

Live trading bot for the GOAT signal strategy, trading **ONDO/USDT:USDT** perpetual futures on Bybit mainnet.

## Architecture

```
goat_live/
├── __init__.py
├── run.py           # Main entry point — the polling loop
├── exchange.py      # ccxt Bybit wrapper: connect, place orders, cancel orders, get position
├── signals.py       # Fetches candles, computes HA + patterns, calls check_goat(), returns signal dict
├── risk.py          # Position sizing ($20 notional → qty), SL/TP price calculation
├── state.py         # Persist last processed candle timestamp + active trade info (JSON file)
├── config.py        # Load .env, build config dict
├── .env.example     # Template with placeholder values
└── README.md        # This file
```

The bot **imports directly from the GOAT repo root** (no code duplication):
- `goat_00_config.py` — direction configs
- `goat_03_patterns.py` — `detect_patterns()` (with fallback to `goat_20_vbt_equity`)
- `goat_04_pivots.py` — `calculate_trade_levels()`
- `goat_05_cases.py` — `check_goat()` (Cases 1 → 2 → 3)

---

## Setup

### 1. Install dependencies

```bash
pip install -r goat_live/requirements.txt
```

### 2. Configure `.env`

```bash
cp goat_live/.env.example goat_live/.env
# Edit goat_live/.env with your Bybit API key and secret
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `BYBIT_API_KEY` | — | **Required** for live trading |
| `BYBIT_API_SECRET` | — | **Required** for live trading |
| `GOAT_SYMBOL` | `ONDO/USDT:USDT` | Bybit linear perp symbol |
| `GOAT_TIMEFRAME` | `1m` | Candle timeframe |
| `GOAT_NOTIONAL_USD` | `20` | Fixed notional per trade in USD |
| `GOAT_RR_RATIO` | `3.0` | Risk-reward ratio |
| `GOAT_PIVOT_LEN` | `2` | Pivot detection lookback |
| `GOAT_LEVERAGE` | `1` | Leverage (set 1 for safety) |
| `GOAT_DRY_RUN` | `true` | **Set false to place real orders** |
| `GOAT_LOG_LEVEL` | `INFO` | Logging verbosity |

### 3. Bybit API key permissions

Create a **read-only + trade** API key on [bybit.com](https://www.bybit.com/app/user/api-management):
- Enable: **Read / Positions / Orders / Trade**
- Disable: **Withdrawals** (for safety)
- Restrict to your server's IP if possible.

---

## Running

Always run from the **repo root** (not inside `goat_live/`):

```bash
# Dry-run first (no real orders):
GOAT_DRY_RUN=true python -m goat_live.run

# Live trading (ensure GOAT_DRY_RUN=false in .env):
python -m goat_live.run
```

Stop with `Ctrl+C` or `SIGTERM` — the bot will cancel open conditional orders and exit cleanly.

---

## How it works

1. Polls Bybit every ~10 seconds.
2. Fetches the last 500 closed 1m candles (drops the forming candle).
3. Computes Heikin-Ashi → runs `detect_patterns()` → runs `check_goat()` for both BULL and BEAR.
4. If a signal fires (Cases C1, C2, or C3):
   - Calls `calculate_trade_levels()` to get `entry / SL / TP`.
   - Sizes the trade at `$20 notional / entry_price` (rounded to lot step).
   - Places: market entry → stop-market SL (reduce-only) → take-profit-market TP (reduce-only).
5. At most **1 open position at a time** — skips signal check if a position exists.
6. Persists state to `goat_live/state.json` so restarts don't double-trade.

---

## Safety notes

- **Start with `GOAT_DRY_RUN=true`** and confirm signals look correct in the logs.
- This bot trades **Bybit mainnet** — real money is at risk.
- Always use a **dedicated API key** with no withdrawal permissions.
- The bot sets leverage on startup; default is `1x` (adjust via `GOAT_LEVERAGE`).
- SL and TP are **reduce-only** orders — they can only close a position, never open a new one.
- The `state.json` file is **not committed to git** (excluded via `.gitignore`).

---

## File: `state.json`

Auto-created at `goat_live/state.json`. Contains:
```json
{
  "last_processed_ts": 1734567890000,
  "open_trade": {
    "side": "BULL",
    "case_label": "Case 1 (LGCR sweep)",
    "entry": 0.7321,
    "sl": 0.6800,
    "tp": 0.8886,
    "qty": 27.3,
    "entry_order_id": "...",
    "sl_order_id": "...",
    "tp_order_id": "...",
    "bar_ts": 1734567890000
  }
}
```
