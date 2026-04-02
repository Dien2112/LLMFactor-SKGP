"""
test.py - Backtest SKGP on fixed test_data

Purpose:
    - Compare skgp(stock_target, date, tweets, history) against ground truth labels
      from known historical data.
    - Doesn't load StockNet full dataset, all data is hardcoded in test_data/.

Run:
    python test.py

Output:
    Per-sample: ticker, date, pred, label, correct/wrong status
    Summary   : ACC, correct / total count, benchmark comparison
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from skgp import skgp, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


TEST_DATA_DIR = Path("./test_data")    # directory containing JSON fixtures
CACHE_DIR     = Path("./cache_test")   # isolated cache for tests


def load_fixtures(data_dir: Path = TEST_DATA_DIR) -> list[dict]:
    """
    Load all *.json from test_data/.
    Each file should follow the schema:
        {
            "stock_target": str,
            "date":         str,
            "label":        int,   # ground truth: 1=rise, 0=fall
            "tweets":       list[str],
            "history":      list[{"date": str, "label": int}],
        }
    """
    fixtures: list[dict] = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        fixtures.append(data)
        logger.debug(f"  Loaded fixture: {json_file.name}")

    logger.info(f"  Loaded {len(fixtures)} test fixtures from {data_dir}")
    return fixtures


def run_backtest(test_data: list[dict]) -> list[dict]:
    """
    Run skgp() on each fixture and compare against ground truth label.

    Returns:
        list[dict] with keys: ticker, date, label, pred, correct, ...
    """
    results: list[dict] = []

    for idx, fix in enumerate(test_data):
        stock = fix["stock_target"]
        date  = fix["date"]
        label = fix["label"]

        logger.info(
            f"\n[{idx + 1}/{len(test_data)}] {stock} {date} "
            f"| GT label = {'rise' if label == 1 else 'fall'}"
        )

        result = skgp(
            stock_target=stock,
            date=date,
            tweets=fix["tweets"],
            history=fix["history"],
            cache_dir=CACHE_DIR,
        )

        pred    = result["pred"]
        correct = int(pred == label)

        logger.info(
            f"  -> Prediction: {result['label_text'].upper()} | "
            f"Ground truth: {'RISE' if label == 1 else 'FALL'} | "
            f"{'CORRECT' if correct else 'WRONG'}"
        )

        results.append({
            "ticker":    stock,
            "date":      date,
            "label":     label,
            "pred":      pred,
            "correct":   correct,
            "factors":   result["factors"].replace("\n", " "),
            "relations": " | ".join(result["relations"]),
            "response":  result["raw_response"].replace("\n", " "),
        })

    return results


def print_summary(results: list[dict]) -> None:
    """Print concise result table."""
    print("\n" + "=" * 60)
    print(f"  BACKTEST SUMMARY - {len(results)} samples")
    print("=" * 60)
    print(f"  {'Ticker':<6} {'Date':<12} {'GT':>5} {'Pred':>5} {'OK?':>5}")
    print("  " + "-" * 38)
    for r in results:
        gt_str   = "rise" if r["label"] == 1 else "fall"
        pred_str = "rise" if r["pred"]  == 1 else "fall"
        ok_str   = "CORRECT" if r["correct"] else "INCORRECT"
        print(f"  {r['ticker']:<6} {r['date']:<12} {gt_str:>5} {pred_str:>5} {ok_str:>5}")
    print("  " + "-" * 38)

    n_correct = sum(r["correct"] for r in results)
    acc       = n_correct / len(results) if results else 0.0

    print(f"\n  Correct : {n_correct} / {len(results)}")
    print(f"  ACC     : {acc:.4f}  ({acc * 100:.1f}%)")
    print()

    # Benchmark reference
    benchmark_acc = 0.6632
    if acc >= benchmark_acc:
        print(f"  Pass / Exceeds benchmark LLMFactor-GPT-4 ({benchmark_acc * 100:.2f}%)")
    else:
        print(
            f"  Fails benchmark LLMFactor-GPT-4 ({benchmark_acc * 100:.2f}%). "
            f"Needs {(benchmark_acc - acc) * 100:.1f}pp more"
        )
    print("=" * 60 + "\n")

def main() -> None:

    fixtures = load_fixtures(TEST_DATA_DIR)
    if not fixtures:
        return

    results = run_backtest(fixtures)

    print_summary(results)

    evaluate(results)



if __name__ == "__main__":
    main()
