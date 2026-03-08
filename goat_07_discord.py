"""
07 — Discord notification sender.
"""

import logging
import aiohttp


async def send_discord_notification(message: str, webhook_url: str):
    """Send a message to Discord via webhook."""
    if not webhook_url:
        logging.error("Webhook URL not set.")
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json={"content": message}) as resp:
                if resp.status in (200, 204):
                    logging.info("Notification sent")
                else:
                    logging.error(f"Discord error status {resp.status}")
    except Exception as e:
        logging.error(f"Error sending notification: {e}")