"""
main.py - Full StockNet Batch Evaluation Pipeline

Runs the complete SKGP pipeline on StockNet dataset:
  1. Load price + tweet data for specified tickers/date range
  2. Build test samples
  3. Run SKGP pipeline (Step 1->2->3) on all samples
  4. Evaluate and save results (ACC, MCC, CSV)

Usage:
    # Run with default StockNet settings (87 tickers, 2015-10-01 to 2016-01-01)
    python main.py

    # Custom tickers and date range
    python main.py --tickers AAPL MSFT GOOG --start 2015-10-01 --end 2016-01-01

    # Custom data and output directories
    python main.py --price-dir data/price/preprocessed --tweet-dir data/tweet/preprocessed

    # Dry run: build samples without calling LLM
    python main.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from skgp import (
    build_test_samples,
    run_pipeline,
    evaluate,
    save_results,
    load_stock_table,
)
from llm.llm import llm as LLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default paths (StockNet standard layout)
DEFAULT_PRICE_DIR = Path("./data/price/preprocessed")
DEFAULT_TWEET_DIR = Path("./data/tweet/preprocessed")
DEFAULT_CACHE_DIR = Path("./cache")
DEFAULT_RESULT_DIR = Path("./results")

# Default StockNet test period (paper Section 4.1)
DEFAULT_TEST_START = "2015-10-01"
DEFAULT_TEST_END = "2016-01-01"


def get_all_tickers(price_dir: Path) -> list[str]:
    """Get all available tickers from price data directory."""
    if not price_dir.exists():
        return []
    return sorted(p.stem for p in price_dir.glob("*.txt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLMFactor SKGP - Stock Movement Prediction Pipeline",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to evaluate (default: all tickers in price-dir)",
    )
    parser.add_argument(
        "--start", default=DEFAULT_TEST_START,
        help=f"Test period start date (default: {DEFAULT_TEST_START})",
    )
    parser.add_argument(
        "--end", default=DEFAULT_TEST_END,
        help=f"Test period end date (default: {DEFAULT_TEST_END})",
    )
    parser.add_argument(
        "--price-dir", type=Path, default=DEFAULT_PRICE_DIR,
        help=f"Price data directory (default: {DEFAULT_PRICE_DIR})",
    )
    parser.add_argument(
        "--tweet-dir", type=Path, default=DEFAULT_TWEET_DIR,
        help=f"Tweet data directory (default: {DEFAULT_TWEET_DIR})",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=DEFAULT_RESULT_DIR,
        help=f"Results output directory (default: {DEFAULT_RESULT_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build samples and show stats without running LLM pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate data directories ──────────────────────────────────────
    if not args.price_dir.exists():
        logger.error(
            f"Price directory not found: {args.price_dir}\n"
            f"Please download the StockNet dataset and place price data at:\n"
            f"  {DEFAULT_PRICE_DIR}"
        )
        sys.exit(1)

    if not args.tweet_dir.exists():
        logger.error(
            f"Tweet directory not found: {args.tweet_dir}\n"
            f"Please download the StockNet dataset and place tweet data at:\n"
            f"  {DEFAULT_TWEET_DIR}"
        )
        sys.exit(1)

    # ── Determine tickers ──────────────────────────────────────────────
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = get_all_tickers(args.price_dir)
        if not tickers:
            logger.error(f"No .txt price files found in {args.price_dir}")
            sys.exit(1)

    logger.info(f"Tickers: {len(tickers)} stocks")
    logger.info(f"Test period: {args.start} to {args.end}")

    # ── Build test samples ─────────────────────────────────────────────
    logger.info("Building test samples...")
    samples = build_test_samples(
        tickers=tickers,
        test_start=args.start,
        test_end=args.end,
        price_dir=args.price_dir,
        tweet_dir=args.tweet_dir,
    )

    if not samples:
        logger.error("No test samples built. Check data directories and date range.")
        sys.exit(1)

    logger.info(f"Built {len(samples)} test samples")

    # ── Dry run: just show stats ───────────────────────────────────────
    if args.dry_run:
        ticker_counts: dict[str, int] = {}
        for s in samples:
            ticker_counts[s["ticker"]] = ticker_counts.get(s["ticker"], 0) + 1
        print(f"\n{'='*50}")
        print(f"  DRY RUN - {len(samples)} samples from {len(ticker_counts)} tickers")
        print(f"{'='*50}")
        for t, c in sorted(ticker_counts.items()):
            print(f"  {t:<6} : {c} samples")
        print(f"{'='*50}\n")
        return

    # ── Run SKGP pipeline ──────────────────────────────────────────────
    logger.info("Initializing LLM...")
    llm_instance = LLM()

    logger.info("Running SKGP pipeline...")
    results = run_pipeline(
        samples=samples,
        llm_instance=llm_instance,
        cache_dir=args.cache_dir,
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    evaluate(results)

    # ── Save results ───────────────────────────────────────────────────
    out_path = args.result_dir / "predictions.csv"
    save_results(results, out_path=out_path)
    logger.info(f"Results saved to {out_path}")

    # ── LLM usage stats ───────────────────────────────────────────────
    usage = llm_instance.manager.usage_summary()
    print(f"\n  LLM Usage:")
    print(f"    API calls   : {usage['api_calls']}")
    print(f"    Cached calls: {usage['cached_calls']}")
    print(f"    Total tokens: {usage['total_tokens']}")
    print()


if __name__ == "__main__":
    main()
