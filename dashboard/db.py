"""
db.py — SQLite helper for GOAT trade dashboard.
"""
import os
import aiosqlite
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "./trades.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id    TEXT PRIMARY KEY,
                symbol      TEXT,
                side        TEXT,
                tf          TEXT,
                case_type   TEXT,
                entry       REAL,
                sl          REAL,
                tp          REAL,
                risk        REAL,
                swept       REAL,
                status      TEXT DEFAULT 'active',
                opened_at   TEXT,
                closed_at   TEXT,
                close_price REAL,
                pnl         REAL,
                max_r       REAL DEFAULT 0,
                be_level    REAL DEFAULT 1.0,
                exchange    TEXT DEFAULT 'Bybit'
            )
        """)
        # Migration: add exchange column to existing databases
        async with db.execute("PRAGMA table_info(trades)") as cur:
            cols = {row[1] async for row in cur}
        if "exchange" not in cols:
            await db.execute("ALTER TABLE trades ADD COLUMN exchange TEXT DEFAULT 'Bybit'")
        await db.commit()


async def insert_trade(trade: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR IGNORE INTO trades
                (trade_id, symbol, side, tf, case_type,
                 entry, sl, tp, risk, swept, status, opened_at, exchange)
            VALUES
                (:trade_id, :symbol, :side, :tf, :case_type,
                 :entry, :sl, :tp, :risk, :swept, :status, :opened_at,
                 :exchange)
        """, trade)
        await db.commit()


async def close_trade(trade_id: str, status: str, close_price: float | None,
                      pnl: float | None, closed_at: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE trades
               SET status = ?, close_price = ?, pnl = ?, closed_at = ?
             WHERE trade_id = ?
        """, (status, close_price, pnl, closed_at, trade_id))
        await db.commit()


async def get_trades(filters: dict | None = None) -> list[dict]:
    """Return list of trade dicts, optionally filtered."""
    clauses, params = [], []
    if filters:
        if filters.get("tf"):
            clauses.append("tf = ?")
            params.append(filters["tf"])
        if filters.get("symbol"):
            clauses.append("symbol = ?")
            params.append(filters["symbol"])
        if filters.get("status"):
            clauses.append("status = ?")
            params.append(filters["status"])
        if filters.get("from"):
            clauses.append("opened_at >= ?")
            params.append(filters["from"])
        if filters.get("to"):
            clauses.append("opened_at <= ?")
            # Append end-of-day time to include the whole "to" date
            to_val = filters["to"]
            if len(to_val) == 10:   # "YYYY-MM-DD" only
                to_val = to_val + " 23:59:59"
            params.append(to_val)
        if filters.get("side"):
            clauses.append("side = ?")
            params.append(filters["side"])
        if filters.get("case_type"):
            clauses.append("case_type = ?")
            params.append(filters["case_type"])
        if filters.get("exchange"):
            clauses.append("exchange = ?")
            params.append(filters["exchange"])

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM trades {where} ORDER BY opened_at DESC"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_active_symbols() -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT symbol FROM trades WHERE status = 'active'"
        ) as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows if r[0]]


async def update_max_r(trade_id: str, max_r: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE trades SET max_r = ? WHERE trade_id = ? AND max_r < ?",
            (max_r, trade_id, max_r),
        )
        await db.commit()
