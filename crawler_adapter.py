"""
crawler_adapter.py — Data fetching bridge for SKGP pipeline.

Responsibility:
    Given (coin_target, date), return:
      - news     : list[str]   — filtered news headlines + summaries for that date
      - history  : list[dict]  — 5 prior daily price sessions (label=0/1)

Usage:
    from crawler_adapter import fetch_data_for_date
    news, history = fetch_data_for_date("BTC", "2026-04-02")
"""

import os
import sys
import logging
import requests
import datetime
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
logger = logging.getLogger(__name__)

from crawler.src.fetch_rss import fetch_feed
from crawler.src.config import DEFAULT_FEEDS

# ---------------------------------------------------------------------------
# KEYWORD MAP: coin ticker -> keywords to filter news
# ---------------------------------------------------------------------------
_KEYWORDS: dict[str, list[str]] = {
    "BTC":   ["btc", "bitcoin"],
    "ETH":   ["eth", "ethereum"],
    "SOL":   ["sol", "solana"],
    "BNB":   ["bnb", "binance coin"],
    "XRP":   ["xrp", "ripple"],
    "ADA":   ["ada", "cardano"],
    "AVAX":  ["avax", "avalanche"],
    "LINK":  ["link", "chainlink"],
    "DOGE":  ["doge", "dogecoin"],
    "MATIC": ["matic", "polygon"],
    "DOT":   ["dot", "polkadot"],
    "UNI":   ["uni", "uniswap"],
}


def _get_keywords(coin_target: str) -> list[str]:
    return _KEYWORDS.get(coin_target.upper(), [coin_target.lower()])


# ---------------------------------------------------------------------------
# INTERNAL: Fetch news (RSS) filtered by coin & date
# ---------------------------------------------------------------------------
def _fetch_news(coin_target: str, date: str, limit: int) -> list[str]:
    """
    Fetch RSS news articles related to coin_target.
    When date is today: returns live feed.
    When date is past: filters articles published on or before that date.
    Returns list of "title. summary" strings.
    """
    target_date = datetime.date.fromisoformat(date)
    today       = datetime.date.today()
    is_past     = target_date < today

    keywords   = _get_keywords(coin_target)
    news_lines: list[str] = []

    for feed_url in DEFAULT_FEEDS:
        if isinstance(feed_url, dict):
            feed_url = feed_url.get("url")
        try:
            articles = fetch_feed(feed_url, timeout=10)
            for article in articles:
                # Date filtering for historical queries
                if is_past and article.published:
                    try:
                        import email.utils
                        pub = email.utils.parsedate_to_datetime(article.published).date()
                        if pub > target_date:
                            continue
                    except Exception:
                        pass  # skip date filter if unparseable

                title   = article.title or ""
                summary = article.summary or ""
                text    = (title + " " + summary).lower()

                if any(kw in text for kw in keywords):
                    line = f"{title}. {summary}".strip()
                    if line not in news_lines:
                        news_lines.append(line)
                    if len(news_lines) >= limit:
                        return news_lines
        except Exception as e:
            logger.warning(f"Feed error [{feed_url}]: {e}")

    return news_lines


# ---------------------------------------------------------------------------
# INTERNAL: Fetch historical price data from CryptoCompare
# ---------------------------------------------------------------------------
def _fetch_price_history(coin_target: str, date: str, window: int) -> list[dict]:
    """
    Fetch 'window' days of price history BEFORE 'date' from CryptoCompare histoday API.
    Returns list of {date, label} dicts sorted oldest -> newest.
    label = 1 if close > open, else 0.
    """
    # Convert target date to Unix timestamp for CryptoCompare 'toTs' parameter
    target_dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    to_ts     = int(target_dt.timestamp())

    url = (
        f"https://min-api.cryptocompare.com/data/histoday"
        f"?fsym={coin_target.upper()}&tsym=USD&limit={window}&toTs={to_ts}"
    )

    api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()

        if data.get("Response") == "Error":
            logger.error(f"CryptoCompare error: {data.get('Message')}")
            return []

        candles = data.get("Data", [])
        history = []
        for candle in candles[-window:]:
            dt_str = datetime.datetime.fromtimestamp(candle["time"]).strftime("%Y-%m-%d")
            label  = 1 if candle["close"] > candle["open"] else 0
            history.append({"date": dt_str, "label": label})

        return history

    except Exception as e:
        logger.error(f"Price history fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def fetch_data_for_date(coin_target: str, date: str, news_limit: int = 15, history_window: int = 5) -> tuple[list[str], list[dict]]:
    logger.info(f"[Adapter] Fetching data for {coin_target} on {date}")

    news    = _fetch_news(coin_target, date, news_limit)
    history = _fetch_price_history(coin_target, date, history_window)

    logger.info(f"[Adapter] -> {len(news)} news, {len(history)} price sessions")
    return news, history
