"""
SKGP.py - Crypto Knowledge Graph Prediction
Core business logic of the project.

Main public function:
    skgp(coin_target, date, tweets, history, ...) -> dict

Pipeline consists of 3 steps (per arXiv 2406.10811v1):
    Step 1 - Relation   : Relationship between target company and related companies
    Step 2 - Factors    : Extract top-K price impact factors from news
    Step 3 - Prediction : Predict rise/fall trend based on factors + relations + price history
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import datetime
from pathlib import Path

import pandas as pd

from llm.llm import llm

logger = logging.getLogger(__name__)

# ==============================================================================
# DEFAULT CONSTANTS
# ==============================================================================

# Dataset paths (CryptoNet standard)
_DEFAULT_DATA_DIR   = Path("")
_DEFAULT_CACHE_DIR  = Path("./cache")
_DEFAULT_RESULT_DIR = Path("./results")

# SKGP hyper-parameters (paper Section 4.4)
WINDOW_SIZE   = 10   # number of historical price sessions to include in prompt
K_FACTORS     = 5    # top-k factors to extract
MAX_TWEETS    = 30   # maximum tweets embedded in prompt
MAX_RELATIONS = 10  # maximum coin_match processed in Step 1 per sample

# Cache file names
_CACHE_STEP1 = "step1_relation.pkl"
_CACHE_STEP2 = "step2_factor.pkl"
_CACHE_STEP3 = "step3_predict.pkl"


# ==============================================================================
# COIN INFO TABLE
# ==============================================================================

def load_coin_table(path: Path = _DEFAULT_DATA_DIR / "CoinTable") -> dict[str, dict]:
    """
    Parse CoinTable file (tab-separated: Sector \\t $TICKER \\t Company).

    Returns:
        {"BTC": {"company": "Bitcoin", "sector": "Layer 1"}, ...}
    """
    info: dict[str, dict] = {}
    if not path.exists():
        logger.warning(f"CoinTable not found at {path}, using empty dict.")
        return info
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            sector, raw_ticker, company = parts[0], parts[1], parts[2]
            ticker = raw_ticker.lstrip("$")
            info[ticker] = {"company": company.strip(), "sector": sector.strip()}
    return info


# Module-level coin info (lazy-loaded once)
_COIN_INFO: dict[str, dict] | None = None


def get_coin_info() -> dict[str, dict]:
    """Returns the coin info table (loaded once, cached in module)."""
    global _COIN_INFO
    if _COIN_INFO is None:
        _COIN_INFO = load_coin_table()
    return _COIN_INFO


# ==============================================================================
# CACHE UTILITIES
# ==============================================================================

def load_cache(path: Path) -> dict:
    """Load cache from .pkl. Returns {} if not found."""
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  Cache loaded: {path.name} ({len(data)} entries)")
        return data
    return {}


def save_cache(data: dict, path: Path) -> None:
    """Write cache to .pkl (atomic via temp file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)


def find_coin_matches(tweets: list[str], target_ticker: str) -> list[str]:
    """
    Find other tickers mentioned in tweets (via cashtag / project name).
    Excludes the target_ticker itself.
    """
    coin_info = get_coin_info()
    combined   = " ".join(tweets).lower()
    tokens     = set(combined.split())
    found: list[str] = []

    for ticker, info in coin_info.items():
        if ticker == target_ticker:
            continue
        # Match "$BTC", "$btc", plain ticker token, or full company name
        cashtag      = f"${ticker.lower()}"
        company_name = info["company"].lower()
        if cashtag in combined or ticker.lower() in tokens or company_name in combined:
            found.append(ticker)

    return found


def step1_get_relation(
    llm:           llm,
    target_ticker: str,
    match_ticker:  str,
    cache:         dict,
) -> str:
    """
    Use LLM to determine the relationship between 2 projects/companies.
    Cache key: (target_ticker, match_ticker) - does not depend on date.
    """
    key = (target_ticker, match_ticker)
    if key in cache:
        llm.cached_calls += 1
        return cache[key]

    stock_info  = get_coin_info()
    target_name = stock_info.get(target_ticker, {}).get("company", target_ticker)
    match_name  = stock_info.get(match_ticker,  {}).get("company", match_ticker)

    prompt = (
        f"Please fill in the blank and return a complete sentence: "
        f"{target_name} and {match_name} are most likely in a ___ relationship."
    )
    response   = llm.generate(prompt)
    cache[key] = response
    return response


# ==============================================================================
# STEP 2 - FACTOR EXTRACTION
# ==============================================================================

def step2_get_factors(
    llm:    llm,
    ticker: str,
    date:   str,
    tweets: list[str],
    cache:  dict,
) -> str:
    """
    Extract top-K price impact factors from news/tweets.
    Cache key: (ticker, date).
    """
    key = (ticker, date)
    if key in cache:
        llm.cached_calls += 1
        return cache[key]

    coin_info = get_coin_info()
    company    = coin_info.get(ticker, {}).get("company", ticker)
    news_text  = " | ".join(tweets[:MAX_TWEETS])

    prompt = (
        f"Please extract the top {K_FACTORS} factors that may affect the crypto price "
        f"of {company} ({ticker}) from the following news:\n{news_text}"
    )
    response   = llm.generate(prompt)
    cache[key] = response
    return response


# ==============================================================================
# STEP 3 - PREDICTION
# ==============================================================================

def _build_time_template(ticker: str, history: list[dict]) -> str:
    """Convert price history to text (TimeTemplate)."""
    coin_info = get_coin_info()
    company    = coin_info.get(ticker, {}).get("company", ticker)
    lines      = [
        f"On {h['date']}, the crypto price of {company} ({ticker}) "
        f"{'rose' if h['label'] == 1 else 'fell'}."
        for h in history
    ]
    return "\n".join(lines)


def step3_get_prediction(
    llm:       llm,
    ticker:    str,
    date:      str,
    factors:   str,
    relations: list[str],
    history:   list[dict],
    cache:     dict,
) -> tuple[int, str]:
    """
    Predict price trend (rise=1 / fall=0).
    Cache key: (ticker, date).

    Returns: (prediction: int, raw_response: str)
    """
    key = (ticker, date)
    if key in cache:
        llm.cached_calls += 1
        raw  = cache[key]
        text = raw.lower()
        import re
        if re.search(r"will\s+(?:\*\*)?rise\b", text):
            pred = 1
        elif re.search(r"will\s+(?:\*\*)?fall\b", text):
            pred = 0
        else:
            pred = 1 if text.count("rise") >= text.count("fall") else 0
        return pred, raw

    coin_info    = get_coin_info()
    company       = coin_info.get(ticker, {}).get("company", ticker)
    time_template = _build_time_template(ticker, history)
    relation_str  = (
        "\n".join(f"- {r}" for r in relations)
        if relations else
        "No related project/company was mentioned in the news."
    )

    prompt = (
        f"Based on the following information, please judge the direction of the "
        f"crypto price as rise or fall, fill in the blank and give reasons.\n\n"
        f"These are the main factors that may affect this crypto's price recently:\n"
        f"{factors}\n\n"
        f"These are the connections between the companies/projects that have appeared in the news:\n"
        f"{relation_str}\n\n"
        f"{time_template}\n"
        f"On {date}, the crypto price of {company} ({ticker}) will ___."
    )
    raw        = llm.generate(prompt, use_strong_model=True)
    cache[key] = raw
    
    text = raw.lower()
    import re
    if re.search(r"will\s+(?:\*\*)?rise\b", text):
        pred = 1
    elif re.search(r"will\s+(?:\*\*)?fall\b", text):
        pred = 0
    else:
        pred = 1 if text.count("rise") >= text.count("fall") else 0
        
    return pred, raw


# ==============================================================================
# MAIN FUNCTION: skgp()
# ==============================================================================

def skgp(
    coin_target: str,
    date:         str,
    tweets:       list[str],
    history:      list[dict],
    cache_dir:    Path | None = None,
) -> dict:
    """
    Run the full SKGP pipeline (Step 1->2->3) for 1 ticker, 1 date.

    Args:
        coin_target  : Ticker to predict (e.g., "BTC").
        date         : Prediction date, format "YYYY-MM-DD".
        tweets       : List of tweets/news for that date.
        history      : Price history of the last WINDOW_SIZE sessions.
                       Each element: {"date": "YYYY-MM-DD", "label": 0 or 1}.
        llm          : GeminiLLMManager instance (if None, use the default global instance).
        cache_dir    : Cache directory (if None, uses ./cache).

    Returns:
        {
            "pred":         0 or 1,          # 1=rise, 0=fall
            "label_text":   "rise"/"fall",
            "factors":      str,             # output from Step 2
            "relations":    list[str],       # output from Step 1
            "raw_response": str,             # raw output from Step 3
        }
    """
    llm_instance = llm() 

    _cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    _cache_dir.mkdir(parents=True, exist_ok=True)

    cache1_path = _cache_dir / _CACHE_STEP1
    cache2_path = _cache_dir / _CACHE_STEP2
    cache3_path = _cache_dir / _CACHE_STEP3

    cache1 = load_cache(cache1_path)
    cache2 = load_cache(cache2_path)
    cache3 = load_cache(cache3_path)

    # Slice to configured maximums before processing
    used_tweets  = tweets[:MAX_TWEETS]
    used_history = history[-WINDOW_SIZE:]

    # -- Step 1: Relations ------------------------------------------------------
    match_tickers = find_coin_matches(used_tweets, coin_target)[:MAX_RELATIONS]
    relations: list[str] = []
    for mt in match_tickers:
        rel = step1_get_relation(llm_instance, coin_target, mt, cache1)
        relations.append(f"{coin_target} & {mt}: {rel}")
    save_cache(cache1, cache1_path)

    # -- Step 2: Factors --------------------------------------------------------
    factors = step2_get_factors(llm_instance, coin_target, date, used_tweets, cache2)
    save_cache(cache2, cache2_path)

    # -- Step 3: Prediction -----------------------------------------------------
    # Pass only WINDOW_SIZE sessions so TimeTemplate is consistent with paper
    pred, raw = step3_get_prediction(
        llm_instance, coin_target, date, factors, relations, used_history, cache3
    )
    save_cache(cache3, cache3_path)

    result = {
        "pred":         pred,
        "label_text":   "rise" if pred == 1 else "fall",
        "factors":      factors,
        "relations":    relations,
        "raw_response": raw,
        # Exact inputs consumed by this skgp() call — used for save_result()
        "_used_input": {
            "tweets":  used_tweets,
            "history": used_history,
        },
    }

    # Cache is a fault-tolerance checkpoint — no longer needed after success
    clear_cache(cache_dir=_cache_dir)

    return result


# ==============================================================================
# SAVE RESULT & CACHE CLEANUP
# ==============================================================================

def save_result(result: dict, coin_target: str, date: str,
                out_dir: Path = _DEFAULT_RESULT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{coin_target}_{date}.json"

    used = result.get("_used_input", {})
    input_payload = {
        "tweets":  used.get("tweets",  []),
        "history": used.get("history", []), 
    }
    with open(f"{out_dir}/{coin_target}_{date}_input.json", "w", encoding="utf-8") as f:
        json.dump(input_payload, f, ensure_ascii=False, indent=2)

    output_payload = {
        "pred":         result["pred"],
        "label_text":   result["label_text"],
        "factors":      result["factors"],
        "relations":    result["relations"],
        "raw_response": result["raw_response"],
    }
    with open(f"{out_dir}/{coin_target}_{date}_output.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    return out_path


def clear_cache(cache_dir: Path | None = None) -> None:
    _dir = cache_dir or _DEFAULT_CACHE_DIR
    for name in [_CACHE_STEP1, _CACHE_STEP2, _CACHE_STEP3]:
        p = _dir / name
        if p.exists():
            p.unlink()
            logger.info(f"  Cache cleared: {p.name}")


