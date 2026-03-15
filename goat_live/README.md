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
| `GOAT_TIMEFRAME` | `1m` | Candle timeframe (see [Changing Timeframe](#changing-timeframe)) |
| `GOAT_NOTIONAL_USD` | `20` | Fixed notional per trade in USD |
| `GOAT_RR_RATIO` | `3.0` | Risk-reward ratio |
| `GOAT_PIVOT_LEN` | `2` | Pivot detection lookback |
| `GOAT_LEVERAGE` | `1` | Leverage (set 1 for safety) |
| `GOAT_DRY_RUN` | `true` | **Set false to place real orders** |
| `GOAT_HEDGE_MODE` | `true` | Enable hedge mode (see [Hedge Mode](#hedge-mode-vs-one-way-mode)) |
| `GOAT_LOG_LEVEL` | `INFO` | Logging verbosity |
| `GOAT_POLL_INTERVAL_SEC` | `15` | Polling frequency in seconds (see [Poll Interval](#poll-interval)) |

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

1. Polls Bybit every ~15 seconds.
2. Fetches the last 500 closed candles (drops the forming candle).
3. Computes Heikin-Ashi → runs `detect_patterns()` → runs `check_goat()` for both BULL and BEAR.
4. If a signal fires (Cases C1, C2, or C3):
   - Calls `calculate_trade_levels()` to get `entry / SL / TP`.
   - Sizes the trade at `$20 notional / entry_price` (rounded to lot step).
   - Places: market entry → stop-market SL (reduce-only) → take-profit-market TP (reduce-only).
5. At most **1 open position at a time** — skips signal check if a position exists.
6. Persists state to `goat_live/state.json` so restarts don't double-trade.

---

## Changing Timeframe

Change the `GOAT_TIMEFRAME` variable in your `.env` to any timeframe Bybit supports:

```env
GOAT_TIMEFRAME=5m   # 5-minute candles
```

Supported values: `1m`, `2m`, `3m`, `5m`, `15m`, `30m`, `1h`, `4h` (and others Bybit supports).

The bot uses this timeframe for both fetching candles and detecting candle closes — no code changes needed.

> **Tip:** When using longer timeframes (e.g. `5m`, `15m`), consider increasing `GOAT_POLL_INTERVAL_SEC`
> to avoid unnecessary API calls between candle closes.

---

## Hedge Mode vs One-Way Mode

Bybit supports two position modes:

| Mode | Behaviour | When to use |
|---|---|---|
| **Hedge** (`GOAT_HEDGE_MODE=true`) | Separate long and short positions simultaneously | Default — most Bybit accounts are in hedge mode |
| **One-way** (`GOAT_HEDGE_MODE=false`) | One position per symbol — a sell order closes a buy | Only if you explicitly switched your account to one-way mode |

> **Important:** If your Bybit account is set to hedge mode, you **must** set `GOAT_HEDGE_MODE=true`.
> Bybit rejects orders without `positionIdx` when the account is in hedge mode.
> The default is `true` because most Bybit accounts use hedge mode.

To switch to one-way mode, set:

```env
GOAT_HEDGE_MODE=false
```

---

## Poll Interval

`GOAT_POLL_INTERVAL_SEC` (default: `15`) controls how often the bot checks for a new candle.

```env
GOAT_POLL_INTERVAL_SEC=15
```

The bot only fetches OHLCV data and places orders when a **new** candle has closed.
Polling faster than the candle interval just means slightly quicker detection — but also means
more API calls, which can trigger Bybit rate limits on busy accounts.

Recommended values:

| Timeframe | Recommended poll interval |
|---|---|
| `1m` | `15` (default) |
| `2m` – `5m` | `20` – `30` |
| `15m` – `1h` | `30` – `60` |

---

## Safety notes

- **Start with `GOAT_DRY_RUN=true`** and confirm signals look correct in the logs.
- This bot trades **Bybit mainnet** — real money is at risk.
- Always use a **dedicated API key** with no withdrawal permissions.
- The bot sets leverage on startup; default is `1x` (adjust via `GOAT_LEVERAGE`).
- SL and TP are **reduce-only** orders — they can only close a position, never open a new one.
- The `state.json` file is **not committed to git** (excluded via `.gitignore`).

---

## Troubleshooting

### `RateLimitExceeded` / Too many visits

The bot makes several API calls per poll cycle. If you see:

```
ccxt.base.errors.RateLimitExceeded: bybit {"retCode":10006,"retMsg":"Too many visits..."}
```

**Fixes:**
- Increase `GOAT_POLL_INTERVAL_SEC` (e.g. `20` or `30`).
- The bot already has `enableRateLimit: true` in the ccxt config (auto-throttles) and retries `fetch_ohlcv` up to 3 times with backoff (5s/10s/15s).
- After any unhandled API error the main loop waits 30 seconds before retrying.

### Timestamp / `recv_window` errors

```
bybit {"retCode":10002,"retMsg":"invalid request, please check your server timestamp or recv_window param..."}
```

This means your system clock is out of sync with Bybit's servers (common on WSL).

**Quick fix:**
```bash
sudo ntpdate time.windows.com
```

**Permanent fix (WSL):**
```bash
sudo apt install systemd-timesyncd -y
sudo timedatectl set-ntp true
```

Or from **PowerShell (Admin)** on the Windows side:
```powershell
wsl --shutdown
```
Then reopen your WSL terminal — the clock resyncs on restart.

The bot also sets `recvWindow: 10000` (10 seconds, double the Bybit default of 5s) to give extra
tolerance for minor clock drift.

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
