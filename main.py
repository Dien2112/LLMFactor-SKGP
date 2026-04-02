import datetime
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from skgp import skgp, save_result
from crawler_adapter import fetch_data_for_date


def main():
    coin_target = "BTC"
    date        = datetime.date.today().isoformat()
    news_limit = 15
    history_window = 5

    print(f"\n{'='*60}")
    print(f"  MarketLens SKGP  |  {coin_target}  |  {date}")
    print(f"{'='*60}\n")

    print("[1/3] Fetching data via crawler adapter...")
    news, history = fetch_data_for_date(coin_target, date, news_limit, history_window)
    print(f"       news={len(news)} articles,  history={len(history)} sessions\n")

    print("[2/3] Running SKGP prediction pipeline...")
    result = skgp(
        coin_target=coin_target,
        date=date,
        tweets=news,
        history=history,
    )

    print("[3/3] Saving result...")
    out_path = save_result(result, coin_target, date)
    print(f"       Saved -> {out_path}\n")

    trend = "RISE" if result["pred"] == 1 else "FALL"
    print(f"{'='*60}")
    print(f"  TARGET  : {coin_target}")
    print(f"  DATE    : {date}")
    print(f"  TREND   : {trend}")
    print(f"{'='*60}")
    print(f"\n[Factors]\n{result['factors']}\n")
    if result["relations"]:
        print(f"[Relations]\n" + "\n".join(result["relations"]) + "\n")
    print(f"[AI Reasoning]\n{result['raw_response']}\n")


if __name__ == "__main__":
    main()
