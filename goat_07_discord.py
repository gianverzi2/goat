"""
07 — Discord notification sender.
"""

import logging
import aiohttp


DASHBOARD_URL = "http://localhost:8080"

# Map webhook URLs to timeframes for the dashboard
_WEBHOOK_TF_MAP = {
    "1471942081177190647": "m30",
    "1472619891168247950": "h4",
    "1472894304249974881": "d",
}


def _guess_tf(webhook_url: str) -> str:
    """Guess timeframe from webhook URL."""
    for key, tf in _WEBHOOK_TF_MAP.items():
        if key in webhook_url:
            return tf
    return "m30"


async def send_discord_notification(message: str, webhook_url: str):
    """Send a message to Discord via webhook, and also to the dashboard."""
    if not webhook_url:
        logging.error("Webhook URL not set.")
        return
    try:
        async with aiohttp.ClientSession() as session:
            # Send to Discord
            async with session.post(webhook_url, json={"content": message}) as resp:
                if resp.status in (200, 204):
                    logging.info("Notification sent")
                else:
                    logging.error(f"Discord error status {resp.status}")

            # Also send to dashboard
            try:
                tf = _guess_tf(webhook_url)
                await session.post(
                    f"{DASHBOARD_URL}/webhook?tf={tf}",
                    data=message,
                    headers={"Content-Type": "text/plain"},
                )
            except Exception:
                pass  # dashboard is optional
    except Exception as e:
        logging.error(f"Error sending notification: {e}")
