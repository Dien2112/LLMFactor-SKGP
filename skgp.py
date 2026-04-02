"""
SKGP.py - Stock Knowledge Graph Prediction
Core business logic of the project.

Main public function:
    skgp(stock_target, date, tweets, history, ...) -> dict

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
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from llm.llm import llm

logger = logging.getLogger(__name__)

# ==============================================================================
# DEFAULT CONSTANTS
# ==============================================================================

# Dataset paths (StockNet standard)
_DEFAULT_DATA_DIR   = Path("./data")
_DEFAULT_PRICE_DIR  = _DEFAULT_DATA_DIR / "price"  / "preprocessed"
_DEFAULT_TWEET_DIR  = _DEFAULT_DATA_DIR / "tweet"  / "preprocessed"
_DEFAULT_CACHE_DIR  = Path("./cache")
_DEFAULT_RESULT_DIR = Path("./results")

# SKGP hyper-parameters (paper Section 4.4)
WINDOW_SIZE   = 5    # number of historical price sessions to include in prompt
K_FACTORS     = 5    # top-k factors to extract
MAX_TWEETS    = 10   # maximum tweets embedded in prompt
MAX_RELATIONS = 3    # maximum stock_match processed in Step 1 per sample

# Cache file names
_CACHE_STEP1 = "step1_relation.pkl"
_CACHE_STEP2 = "step2_factor.pkl"
_CACHE_STEP3 = "step3_predict.pkl"


# ==============================================================================
# STOCK INFO TABLE
# ==============================================================================

def load_stock_table(path: Path = _DEFAULT_DATA_DIR / "StockTable") -> dict[str, dict]:
    """
    Parse StockTable file (tab-separated: Sector \\t $TICKER \\t Company).

    Returns:
        {"AAPL": {"company": "Apple Inc.", "sector": "Consumer Goods"}, ...}
    """
    info: dict[str, dict] = {}
    if not path.exists():
        logger.warning(f"StockTable not found at {path}, using empty dict.")
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


# Module-level stock info (lazy-loaded once)
_STOCK_INFO: dict[str, dict] | None = None


def get_stock_info() -> dict[str, dict]:
    """Returns the stock info table (loaded once, cached in module)."""
    global _STOCK_INFO
    if _STOCK_INFO is None:
        _STOCK_INFO = load_stock_table()
    return _STOCK_INFO


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


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_price(ticker: str, price_dir: Path = _DEFAULT_PRICE_DIR) -> pd.DataFrame | None:
    """
    Load price data from preprocessed .txt (StockNet format).

    Columns (tab-sep, no header):
        date | ret_open | ret_high | ret_low | ret_close | price_change | volume

    Label: ret_close > 0 -> rise (1), else -> fall (0).
    """
    path = price_dir / f"{ticker}.txt"
    if not path.exists():
        logger.warning(f"  Price file not found: {path}")
        return None

    cols = ["date", "ret_open", "ret_high", "ret_low",
            "ret_close", "price_change", "volume"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    df["date"]  = df["date"].astype(str).str.strip()
    df          = df.sort_values("date").reset_index(drop=True)
    df["label"] = (df["ret_close"] > 0).astype(int)
    return df


def load_tweets(ticker: str, date: str,
                tweet_dir: Path = _DEFAULT_TWEET_DIR) -> list[str]:
    """
    Load tweets for the ticker on a specific date.
    Each line is a JSON with key "text" containing a list of tokens.

    Returns: list of string tweets (tokens joined).
    """
    tweet_file = tweet_dir / ticker / date
    if not tweet_file.exists():
        return []

    texts: list[str] = []
    with open(tweet_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj    = json.loads(line)
                tokens = obj.get("text", [])
                text   = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def build_test_samples(
    tickers:    list[str],
    test_start: str = "2015-10-01",
    test_end:   str = "2016-01-01",
    price_dir:  Path = _DEFAULT_PRICE_DIR,
    tweet_dir:  Path = _DEFAULT_TWEET_DIR,
) -> list[dict]:
    """
    Create a list of test samples from price + tweet data.

    Each sample:
        {
            "ticker":  str,
            "date":    "YYYY-MM-DD",
            "history": [{"date": str, "label": int}, ...],   # WINDOW_SIZE sessions
            "tweets":  [str, ...],
            "label":   int,   # 1=rise, 0=fall
        }
    """
    samples: list[dict] = []

    for ticker in tickers:
        df = load_price(ticker, price_dir)
        if df is None or len(df) < WINDOW_SIZE + 1:
            logger.warning(f"  [{ticker}] Skipped: insufficient price data")
            continue

        for i in range(WINDOW_SIZE, len(df)):
            row  = df.iloc[i]
            date = row["date"]

            if not (test_start <= date <= test_end):
                continue

            tweets = load_tweets(ticker, date, tweet_dir)
            if not tweets:
                continue

            history = [
                {"date": df.iloc[j]["date"], "label": int(df.iloc[j]["label"])}
                for j in range(i - WINDOW_SIZE, i)
            ]
            samples.append({
                "ticker":  ticker,
                "date":    date,
                "history": history,
                "tweets":  tweets,
                "label":   int(row["label"]),
            })

    logger.info(f"  Total test samples: {len(samples)} (from {len(tickers)} tickers)")
    return samples


# ==============================================================================
# STEP 1 - RELATION (Background Knowledge)
# ==============================================================================

def find_stock_matches(tweets: list[str], target_ticker: str) -> list[str]:
    """
    Find other tickers mentioned in tweets (via cashtag / company name).
    Excludes the target_ticker itself.
    """
    stock_info = get_stock_info()
    combined   = " ".join(tweets).lower()
    tokens     = set(combined.split())
    found: list[str] = []

    for ticker, info in stock_info.items():
        if ticker == target_ticker:
            continue
        cashtag      = f"$ {ticker.lower()}"
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
    Use LLM to determine the relationship between 2 companies.
    Cache key: (target_ticker, match_ticker) - does not depend on date.
    """
    key = (target_ticker, match_ticker)
    if key in cache:
        llm.cached_calls += 1
        return cache[key]

    stock_info  = get_stock_info()
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

    stock_info = get_stock_info()
    company    = stock_info.get(ticker, {}).get("company", ticker)
    news_text  = " | ".join(tweets[:MAX_TWEETS])

    prompt = (
        f"Please extract the top {K_FACTORS} factors that may affect the stock price "
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
    stock_info = get_stock_info()
    company    = stock_info.get(ticker, {}).get("company", ticker)
    lines      = [
        f"On {h['date']}, the stock price of {company} ({ticker}) "
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
        pred = 1 if "rise" in raw.lower() else 0
        return pred, raw

    stock_info    = get_stock_info()
    company       = stock_info.get(ticker, {}).get("company", ticker)
    time_template = _build_time_template(ticker, history)
    relation_str  = (
        "\n".join(f"- {r}" for r in relations)
        if relations else
        "No related company was mentioned in the news."
    )

    prompt = (
        f"Based on the following information, please judge the direction of the "
        f"stock price as rise or fall, fill in the blank and give reasons.\n\n"
        f"These are the main factors that may affect this stock's price recently:\n"
        f"{factors}\n\n"
        f"These are the connections between the companies that have appeared in the news:\n"
        f"{relation_str}\n\n"
        f"{time_template}\n"
        f"On {date}, the stock price of {company} ({ticker}) will ___."
    )
    raw        = llm.generate(prompt, use_strong_model=True)
    cache[key] = raw
    pred       = 1 if "rise" in raw.lower() else 0
    return pred, raw


# ==============================================================================
# MAIN FUNCTION: skgp()
# ==============================================================================

def skgp(
    stock_target: str,
    date:         str,
    tweets:       list[str],
    history:      list[dict],
    cache_dir:    Path | None = None,
) -> dict:
    """
    Run the full SKGP pipeline (Step 1->2->3) for 1 ticker, 1 date.

    Args:
        stock_target : Ticker to predict (e.g., "AAPL").
        date         : Prediction date, format "YYYY-MM-DD".
        tweets       : List of tweets/news for that date.
        history      : Price history of the last WINDOW_SIZE sessions.
                       Each element: {"date": "YYYY-MM-DD", "label": 0 or 1}.
        llm          : GeminiLLM instance (if None, use the default global instance).
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

    # -- Step 1: Relations ------------------------------------------------------
    match_tickers = find_stock_matches(tweets, stock_target)[:MAX_RELATIONS]
    relations: list[str] = []
    for mt in match_tickers:
        rel = step1_get_relation(llm_instance, stock_target, mt, cache1)
        relations.append(f"{stock_target} & {mt}: {rel}")
    save_cache(cache1, cache1_path)

    # -- Step 2: Factors --------------------------------------------------------
    factors = step2_get_factors(llm_instance, stock_target, date, tweets, cache2)
    save_cache(cache2, cache2_path)

    # -- Step 3: Prediction -----------------------------------------------------
    pred, raw = step3_get_prediction(
        llm_instance, stock_target, date, factors, relations, history, cache3
    )
    save_cache(cache3, cache3_path)

    return {
        "pred":         pred,
        "label_text":   "rise" if pred == 1 else "fall",
        "factors":      factors,
        "relations":    relations,
        "raw_response": raw,
    }


# ==============================================================================
# FULL PIPELINE (batch mode - used in main.py)
# ==============================================================================

def run_pipeline(
    samples:   list[dict],
    llm:       GeminiLLM,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> list[dict]:
    """
    Run SKGP pipeline on the full test sample list.
    Supports resume: caching skips API calls for previously processed samples.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    _DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading caches...")
    cache1 = load_cache(cache_dir / _CACHE_STEP1)
    cache2 = load_cache(cache_dir / _CACHE_STEP2)
    cache3 = load_cache(cache_dir / _CACHE_STEP3)

    results: list[dict] = []
    n = len(samples)

    for idx, sample in enumerate(samples):
        ticker  = sample["ticker"]
        date    = sample["date"]
        tweets  = sample["tweets"]
        history = sample["history"]
        label   = sample["label"]

        logger.info(
            f"[{idx + 1}/{n}] {ticker} {date} | "
            f"label={'rise' if label == 1 else 'fall'}"
        )

        # Step 1
        match_tickers = find_stock_matches(tweets, ticker)[:MAX_RELATIONS]
        relations: list[str] = []
        for mt in match_tickers:
            rel = step1_get_relation(llm, ticker, mt, cache1)
            relations.append(f"{ticker} & {mt}: {rel}")
        save_cache(cache1, cache_dir / _CACHE_STEP1)
        logger.info(f"  Step1: {len(relations)} relations")

        # Step 2
        factors      = step2_get_factors(llm, ticker, date, tweets, cache2)
        step2_cached = (ticker, date) in cache2
        save_cache(cache2, cache_dir / _CACHE_STEP2)
        logger.info(f"  Step2: factors extracted ({'cached' if step2_cached else 'new'})")

        # Step 3
        pred, raw    = step3_get_prediction(llm, ticker, date, factors, relations, history, cache3)
        step3_cached = (ticker, date) in cache3
        save_cache(cache3, cache_dir / _CACHE_STEP3)

        correct = int(pred == label)
        logger.info(
            f"  Step3: pred={'rise' if pred == 1 else 'fall'} | "
            f"label={'rise' if label == 1 else 'fall'} | "
            f"{'✓ CORRECT' if correct else '✗ WRONG'}"
        )

        results.append({
            "ticker":    ticker,
            "date":      date,
            "label":     label,
            "pred":      pred,
            "correct":   correct,
            "factors":   factors.replace("\n", " "),
            "relations": " | ".join(relations),
            "response":  raw.replace("\n", " "),
        })

        if (idx + 1) % 10 == 0:
            _log_running_stats(results)

    return results


def _log_running_stats(results: list[dict]) -> None:
    if not results:
        return
    acc = sum(r["correct"] for r in results) / len(results)
    logger.info(f"  -- Running ACC ({len(results)} samples): {acc:.3f} ({acc * 100:.1f}%) --")


# ==============================================================================
# EVALUATE & SAVE
# ==============================================================================

def evaluate(results: list[dict]) -> None:
    """Print ACC and MCC result table."""
    if not results:
        print("No results to evaluate.")
        return

    y_true = [r["label"] for r in results]
    y_pred = [r["pred"]  for r in results]

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 54)
    print(f"  RESULTS - {len(results)} test samples")
    print("=" * 54)
    print(f"  Accuracy (ACC) : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  MCC            : {mcc:.4f}")
    print("  Confusion Matrix:")
    print("      Predicted Fall  Predicted Rise")
    if cm.shape == (2, 2):
        print(f"  Actual Fall   TN={cm[0][0]:<6}  FP={cm[0][1]}")
        print(f"  Actual Rise   FN={cm[1][0]:<6}  TP={cm[1][1]}")
    print("=" * 54)
    print("\n  Benchmark (StockNet - paper Table 2):")
    print("    LLMFactor_GPT-4 : ACC=66.32  MCC=0.238")
    print(f"    Your run         : ACC={acc * 100:.2f}  MCC={mcc:.3f}")
    print()


def save_results(results: list[dict], out_path: Path = _DEFAULT_RESULT_DIR / "predictions.csv") -> None:
    """Save prediction results to CSV."""
    if not results:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "date", "label", "pred", "correct",
                  "factors", "relations", "response"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info(f"  Results saved to: {out_path}")
