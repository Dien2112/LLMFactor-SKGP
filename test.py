import datetime
from skgp import skgp
from crawler_adapter import fetch_data_for_date


def main():
    coin_target    = "BTC"
    news_limit     = 100
    history_window = 11# context sessions; +1 fetched below as ground truth

    records = []

    for index in range(1, 16):
        date = (datetime.date.today() - datetime.timedelta(days=index * history_window)).isoformat()
        print(f"\n{'='*60}")
        print(f"  {coin_target}  |  {date}")
        print(f"{'='*60}")

        news, history = fetch_data_for_date(coin_target, date, news_limit, history_window + 1)
        if len(history) < history_window + 1:
            break

        # Last session = ground truth for `date`; rest = context
        ground_truth = history[-1]["label"]
        history      = history[:-1]

        result = skgp(coin_target=coin_target, date=date, tweets=news, history=history)
        pred   = result["pred"]

        correct = pred == ground_truth
        print(f"  Pred={'RISE' if pred==1 else 'FALL'}  |  GT={'RISE' if ground_truth==1 else 'FALL'}  |  {'CORRECT' if correct else 'WRONG'}")
        records.append({"pred": pred, "label": ground_truth, "correct": int(correct)})

    # Summary
    n         = len(records)
    n_correct = sum(r["correct"] for r in records)
    accuracy  = n_correct / n if n else 0

    tp = sum(1 for r in records if r["pred"] == 1 and r["label"] == 1)
    fp = sum(1 for r in records if r["pred"] == 1 and r["label"] == 0)
    fn = sum(1 for r in records if r["pred"] == 0 and r["label"] == 1)
    tn = n - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mcc       = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5    

    print(f"\n{'='*60}")
    print(f"  Accuracy  : {accuracy:.2%}  ({n_correct}/{n})")
    print(f"  Precision : {precision:.2%}")
    print(f"  Recall    : {recall:.2%}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
