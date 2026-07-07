# 🐐 GOAT Trade Dashboard

A lightweight web dashboard that displays live GOAT trades from Discord webhook messages.

## Architecture

```
GOAT bot → Discord webhook → Dashboard FastAPI backend → Browser
                                    ↓
                              Bybit price polling (ccxt, no API key)
                              SQLite database (trades.db)
```

## Features

- **Live trade cards** with horizontal SL—Entry—TP price bars
- **Live price dot** — current price moves along the bar every 5 seconds (Bybit public API)
- **1R milestone marker** — yellow marker on the bar; turns into a purple "SL→BE" indicator when peak R ≥ 1.0
- **Bar chart** of closed trade results with cumulative R line
- **Filter bar** — coin, timeframe, side, status, case, date range
- **Stats row** — Active / WIN / LOSS / BE / WR% / Total P&L (R)
- **Dark theme** — GitHub-style dark palette

## Quick Start (local)

```bash
cd dashboard
pip install -r requirements.txt
cp .env.example .env          # optional — edit port/path
uvicorn app:app --reload --port 8080
# Open http://localhost:8080

# Or run directly (uses HOST/PORT from .env, defaults HOST=0.0.0.0 PORT=8080)
python app.py
```

## Deploy on AWS EC2 (Ubuntu)

```bash
# On your server
cd ~/DigitalOcean/goat
git pull origin main

# Install dependencies
pip3 install -r dashboard/requirements.txt

# Run in foreground
cd dashboard
uvicorn app:app --host 0.0.0.0 --port 8080

# Or run in background
nohup uvicorn app:app --host 0.0.0.0 --port 8080 > dashboard.log 2>&1 &

# Open firewall
sudo ufw allow 8080
```

Access at: `http://<aws-ip>:8080`

## Webhook endpoint

```
POST /webhook?tf=m30
Content-Type: text/plain

<raw Discord message text>
```

Supported `tf` values: `m30`, `h4`, `d`

### Integration with the GOAT bot

In `goat_07_discord.py` (or wherever `send_discord_notification` is called), add a POST
to the dashboard after each Discord message:

```python
import aiohttp

async def send_to_dashboard(message: str, timeframe: str = "m30",
                             dashboard_url: str = "http://localhost:8080"):
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(
                f"{dashboard_url}/webhook?tf={timeframe}",
                data=message,
                headers={"Content-Type": "text/plain"},
            )
    except Exception:
        pass  # dashboard is optional
```

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook?tf=m30` | Receive raw Discord message |
| `GET`  | `/api/trades` | All trades (supports filters: `tf`, `symbol`, `status`, `side`, `case_type`, `from`, `to`) |
| `GET`  | `/api/prices` | Current prices for active symbols |
| `GET`  | `/` | Dashboard UI |

## Files

```
dashboard/
├── app.py           FastAPI application
├── parser.py        Discord open/close message parser
├── db.py            SQLite helper
├── price_feed.py    Bybit price poller (ccxt)
├── static/
│   └── index.html   Single-page dashboard
├── requirements.txt
├── .env.example
└── README.md
```

## BE / 1R milestone

When a trade's peak favorable R (`max_r`) reaches **1.0 R**, the SL marker on the
price bar changes colour from red to purple and shows **"SL→BE"** — a reminder to
manually move your stop-loss to breakeven on the exchange. The dashboard is
read-only and does **not** move any stops automatically.
