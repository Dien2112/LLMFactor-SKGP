# Thiết Kế Tổng Thể Đồ Án: LLMFactor – Dự Đoán Xu Hướng Giá Cổ Phiếu bằng SKGP

> **Paper gốc:** *LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction* (arXiv: 2406.10811v1)  
> **Tên project:** SKGP (Sequential Knowledge-Guided Prompting)  
> **Phạm vi:** Reproduce kết quả paper trên 4 benchmark dataset gốc. **Không mở rộng thêm.**

---

## 1. Tổng Quan Bài Toán

**Bài toán:** Phân loại nhị phân — dự đoán giá đóng cửa ngày `t+1` của cổ phiếu sẽ **tăng (rise=1)** hay **giảm (fall=0)** so với ngày `t`.

**Đầu vào mỗi mẫu:**
- `stock_target`: ticker cổ phiếu cần dự đoán
- `news_target`: tin tức/tweets ngày `date_target`
- `P̂ = {P̂₁, P̂₂, ..., P̂₅}`: chuỗi biến động giá 5 phiên liền trước (window size `t=5`)

**Đầu ra:** `P̂_{t+1} ∈ {rise, fall}`

---

## 2. Dataset

### 2.1 Bốn dataset benchmark (theo paper, Table 1)

| Dataset | Thị trường | Số cổ phiếu | Khoảng thời gian | Loại dữ liệu | Tổng mẫu |
|---|---|---|---|---|---|
| **StockNet** | US (NYSE/NASDAQ) | 87 | 2014-01-01 → 2016-01-01 | Giá + Tweets | 19,318 |
| **CMIN-US** | US (top 110) | 110 | 2018-01-01 → 2021-12-31 | Giá + Tweets | 83,553 |
| **CMIN-CN** | CN (CSI 300) | 300 | 2018-01-01 → 2021-12-31 | Giá + Tweets | 198,781 |
| **EDT** | US | 4,228 | 2020-03-01 → 2021-05-06 | Giá + News articles | 54,080 |

> **Lưu ý EDT:** Không có historical price sequence → Step 1 (Relation) và TimeTemplate **không dùng** cho EDT.

### 2.2 Nguồn tải dataset

| Dataset | Link GitHub | License |
|---|---|---|
| StockNet | https://github.com/yumoxu/stocknet-dataset | MIT |
| CMIN-US / CMIN-CN | https://github.com/wuhuizhe/CHRNN | - |
| EDT | https://github.com/Zhihan1996/TradeTheEvent | - |

### 2.3 Phân chia Train / Validation / Test

Paper không nêu tỷ lệ cụ thể, nhưng bám theo convention của từng dataset gốc:

| Dataset | Train | Validation | Test | Cơ sở |
|---|---|---|---|---|
| StockNet | 2014-01-01 → 2015-07-31 | 2015-08-01 → 2015-09-30 | 2015-10-01 → 2016-01-01 | Xu & Cohen (2018) |
| CMIN-US | 2018-01-01 → 2020-06-30 | 2020-07-01 → 2020-12-31 | 2021-01-01 → 2021-12-31 | Luo et al. (2023) |
| CMIN-CN | 2018-01-01 → 2020-06-30 | 2020-07-01 → 2020-12-31 | 2021-01-01 → 2021-12-31 | Luo et al. (2023) |
| EDT | 2020-03-01 → 2020-12-31 | - | 2021-01-01 → 2021-05-06 | Zhou et al. (2021) |

**Số mẫu test ước tính** (dựa theo split trên):

| Dataset | Tổng mẫu | Tỷ lệ test | **Mẫu test ước tính** |
|---|---|---|---|
| StockNet | 19,318 | ~17% (3/18 tháng) | **~3,300** |
| CMIN-US | 83,553 | ~25% (12/48 tháng) | **~20,900** |
| CMIN-CN | 198,781 | ~25% (12/48 tháng) | **~49,700** |
| EDT | 54,080 | ~28% (5/14 tháng) | **~14,700** |

---

## 3. Kiến Trúc SKGP – 3 Bước

### 3.1 Sơ đồ pipeline

```
Mỗi mẫu (stock_target, date_target, news_target, hist_prices):

news_target + stock_list  ─► [Step 1] Relation prompt ─► relation
news_target               ─► [Step 2] Factor prompt   ─► factors[]
relation + factors + hist ─► [Step 3] Price prompt    ─► prediction (rise/fall)
```

### 3.2 Chi tiết 3 bước (theo paper)

#### Step 1 – Matching & Background Knowledge (RelationTemplate)
```
INPUT: stock_target, stock_match (từ stock_list ∩ news)
PROMPT: "Please fill in the blank and return a complete sentence:
         {stock_target} and {stock_match} are most likely in a ___ relationship."
OUTPUT: relation  (e.g., "competitor relationship")
```
- Gọi 1 lần cho **mỗi cặp** `(stock_target, stock_match)` tìm được trong tin tức
- Nếu không có stock_match → bỏ qua, `relation = ""`

#### Step 2 – Factor Extraction (FactorTemplate)
```
INPUT: stock_target, news_target
PROMPT: "Please extract the top {k} factors that may affect the stock price
         of {stock_target} from the following news: {news_target}"
OUTPUT: factor_1, factor_2, ..., factor_k  (k=5)
```
- Gọi **1 lần** cho mỗi mẫu

#### Step 3 – Price Movement Prediction (PriceTemplate)
```
INPUT: relation, factors, TimeTemplate (5 ngày lịch sử), date_target
PROMPT: "Based on the following information, please judge the direction of
         the stock price from rise/fall, fill in the blank and give reasons.
         
         These are the main factors: {factors}
         These are company connections: {relation}
         On {date_{i-5}}, the stock price of {stock_target} rose/fell.
         On {date_{i-4}}, the stock price of {stock_target} rose/fell.
         On {date_{i-3}}, the stock price of {stock_target} rose/fell.
         On {date_{i-2}}, the stock price of {stock_target} rose/fell.
         On {date_{i-1}}, the stock price of {stock_target} rose/fell.
         On {date_i}, the stock price of {stock_target} will ___."
OUTPUT: "rise" hoặc "fall" + lý do
```
- Gọi **1 lần** cho mỗi mẫu

### 3.3 Tham số thực nghiệm (theo paper, Section 4.4)
| Tham số | Giá trị |
|---|---|
| Window size `t` | 5 |
| Số factors `k` | 5 |
| Batch size (GPT) | 5 mẫu/batch |
| Batch size (BERT) | 64 |
| LLM models dùng | gpt-3.5-turbo-1106, gpt-4, gpt-4-1106-preview |

---

## 4. Pseudocode Triển Khai

### Phase 0 – Tiền xử lý dữ liệu

```python
def prepare_data(dataset_path, ticker_list, split="test"):
    """Load và tạo mẫu theo đúng convention của paper gốc."""
    prices, news = {}, {}
    for ticker in ticker_list:
        df = pd.read_csv(f"{dataset_path}/price/{ticker}.csv")
        # Label: 1 nếu close[t] > close[t-1], 0 ngược lại
        df['movement'] = (df['close'].diff() > 0).astype(int)
        prices[ticker] = df
        news[ticker] = load_daily_news(f"{dataset_path}/tweet/{ticker}/")

    stock_list = pd.read_csv(f"{dataset_path}/stock_list.csv")
    # Có thể là file riêng hoặc lấy từ NYSE/NASDAQ listing

    samples = []
    for ticker in ticker_list:
        df = prices[ticker]
        for i in range(5, len(df)):
            date = df.iloc[i]['date']
            if not in_split(date, split):
                continue
            if not news[ticker].get(date):
                continue  # Bỏ ngày không có tin tức
            samples.append({
                'ticker':   ticker,
                'date':     date,
                'history':  df.iloc[i-5:i][['date','movement']].to_dict('records'),
                'news':     news[ticker][date],   # list of tweets/articles
                'label':    df.iloc[i]['movement']
            })
    return samples, stock_list


def in_split(date, split, dataset="stocknet"):
    """Xác định date thuộc tập train/val/test."""
    splits = {
        "stocknet": {
            "train": ("2014-01-01", "2015-07-31"),
            "val":   ("2015-08-01", "2015-09-30"),
            "test":  ("2015-10-01", "2016-01-01"),
        },
        ...
    }
    lo, hi = splits[dataset][split]
    return lo <= date <= hi
```

### Phase 1 – SKGP Inference (3 bước)

```python
def run_skgp_pipeline(test_samples, stock_list, llm_client):
    results = []
    for sample in test_samples:
        # ── Step 1: Relation ──────────────────────────────────────
        matched = match_stocks_in_news(sample['news'], stock_list)
        # matched: list các stock xuất hiện trong tin (khác stock_target)
        relations = []
        for sm in matched:
            prompt_relation = (
                f"Please fill in the blank and return a complete sentence: "
                f"{sample['ticker']} and {sm['company']} "
                f"are most likely in a ___ relationship."
            )
            rel = llm_client.generate(prompt_relation)  # 1 API call
            relations.append(rel)
        relation_str = "; ".join(relations) if relations else ""

        # ── Step 2: Factor Extraction ─────────────────────────────
        news_concat = " ".join(sample['news'])  # ghép tất cả tweets ngày đó
        prompt_factor = (
            f"Please extract the top 5 factors that may affect the stock price "
            f"of {sample['ticker']} from the following news: {news_concat}"
        )
        factors_raw = llm_client.generate(prompt_factor)  # 1 API call
        factors_str = parse_factors(factors_raw)

        # ── Step 3: Prediction ────────────────────────────────────
        time_template = ""
        for h in sample['history']:
            mv = "rose" if h['movement'] == 1 else "fell"
            time_template += f"On {h['date']}, the stock price of {sample['ticker']} {mv}.\n"

        prompt_predict = (
            f"Based on the following information, please judge the direction of "
            f"the stock price from rise/fall, fill in the blank and give reasons.\n"
            f"These are the main factors that may affect this stock's price recently: {factors_str}\n"
            f"These are the connections between the companies that have appeared in the news: {relation_str}\n"
            f"{time_template}"
            f"On {sample['date']}, the stock price of {sample['ticker']} will ___."
        )
        response = llm_client.generate(prompt_predict)  # 1 API call
        pred = 1 if "rise" in response.lower() else 0

        results.append({
            'ticker':     sample['ticker'],
            'date':       sample['date'],
            'prediction': pred,
            'label':      sample['label'],
            'reason':     response,
        })
    return results
```

### Phase 2 – Đánh giá (Evaluation)

```python
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

def evaluate(results, dataset_name=""):
    y_true = [r['label']      for r in results]
    y_pred = [r['prediction'] for r in results]

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    print(f"[{dataset_name}] ACC={acc:.4f} | MCC={mcc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return acc, mcc


def ablation_study(test_samples, stock_list, llm_client):
    """3 layers như trong paper Table 3:
       Layer 1 - Price only
       Layer 2 - Price + Factor
       Layer 3 - Price + Factor + Relation  (Full SKGP)
    """
    for config in [
        ("Price only",            use_relation=False, use_factor=False),
        ("Price + Factor",        use_relation=False, use_factor=True),
        ("Price + Factor + Rel.", use_relation=True,  use_factor=True),
    ]:
        name, use_rel, use_fac = config
        results = run_skgp_pipeline(
            test_samples, stock_list, llm_client,
            use_relation=use_rel, use_factor=use_fac
        )
        acc, mcc = evaluate(results, name)
```

---

## 5. Baseline So Sánh (theo paper)

Paper so sánh với 3 nhóm baseline. **Không cần implement từ đầu**, dùng thư viện có sẵn.

### 5.1 Keyphrase-based (dùng `pke` toolkit)

| Baseline | Mô tả | Thư viện |
|---|---|---|
| PromptRank (ACL 2023) | Unsupervised, dùng PLM + prompt | `pke` |
| KeyBERT | BERT embeddings + cosine similarity | `keybert` |
| YAKE | Statistical keyword extraction | `yake` |
| TextRank | Graph-based co-occurrence | `pke` / `pytextrank` |
| TopicRank | Topic-clustered ranking | `pke` |
| SingleRank | TextRank + weighted edges | `pke` |
| TFIDF | Classical term frequency | `sklearn` |

**Cách tính score cho keyphrase-based (theo paper eq.3):**
```python
# Chia train set thành 2 nhóm (rise/fall)
# Trích keyphrases cho từng nhóm → POS_keyphrases, NEG_keyphrases
# Score(text) = sigmoid(count(POS ∩ text) - count(NEG ∩ text))
```

### 5.2 Sentiment-based

| Baseline | HuggingFace ID | Ngôn ngữ |
|---|---|---|
| FinBERT | `ProsusAI/finbert` | EN |
| FinBERT (CN) | `bardsai/finance-sentiment-zh-base` | CN |
| RoBERTa | `soleimanian/financial-roberta-large-sentiment` | EN |
| RoBERTa (CN) | `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` | CN |
| FinGPT | `FinGPT/fingpt-sentiment_llama2-13b_lora` | EN |
| FinGPT (CN) | `oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT` | CN |
| EDT | `bilevel BERT event detection` | EN |
| GPT-3.5/4/4-turbo | OpenAI API, zero-shot sentiment | EN |

### 5.3 Time-based

| Baseline | Paper | Ghi chú |
|---|---|---|
| StockNet | ACL 2018 | Deep generative model, code tại GitHub gốc |
| CMIN | ACL 2023 | Causality-guided network, code tại GitHub gốc |

> EDT không áp dụng time-based (không có historical price sequence).

---

## 6. Yêu Cầu Phần Cứng & Phần Mềm

### 6.1 Phần mềm

```
Python >= 3.10
pandas >= 2.0
scikit-learn >= 1.3        # evaluate metrics
openai >= 1.0              # GPT API
google-generativeai        # Gemini API (thay thế)
pke                        # keyphrase toolkit (EN)
pke_zh                     # keyphrase toolkit (CN, github: shibing624/pke_zh)
keybert
yake
torch >= 2.0               # cho các baseline BERT/GPT
transformers >= 4.38
```

### 6.2 Phần cứng

| Task | VRAM | RAM | GPU cần |
|---|---|---|---|
| SKGP inference (GPT/Gemini API) | 0 | 8 GB | Không cần |
| Baseline BERT (FinBERT, RoBERTa) inference | 4 GB | 16 GB | GTX 1650 trở lên |
| Baseline FinGPT-13B inference | 24 GB | 64 GB | A100 hoặc 2× RTX 3090 |
| StockNet / CMIN re-training | 4 GB | 16 GB | GTX 1650 trở lên |

> Paper dùng **NVIDIA RTX A6000** (48 GB VRAM) để chạy tất cả baselines.  
> Với máy 4 GB VRAM: chỉ chạy được FinBERT/RoBERTa inference + SKGP qua API.  
> FinGPT-13B cần thuê cloud (Colab Pro hoặc RunPod).

---

## 7. Ước Tính Số API Calls & Tokens cho Phần Test

> Đây là phần **quan trọng nhất** để lên kế hoạch chi phí / thời gian inference.

### 7.1 Giả định (dựa theo paper)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Window size `t` | 5 phiên | Paper Section 4.4 |
| Factors `k` | 5 | Paper Section 4.4 |
| Số stock_match trung bình trong 1 tin | ~1 | Ước tính (tweets ngắn, ít đề cập) |
| %mẫu có ít nhất 1 stock_match | ~60% | Ước tính thực tế |
| Avg Step-1 calls per sample | ~0.6 | 60% × 1 match |

### 7.2 Số tokens mỗi bước (ước tính)

#### Step 1 – RelationTemplate
```
Input:  "Please fill in the blank and return a complete sentence:
         {stock_target} and {stock_match} are most likely in a ___ relationship."
         → ~35 tokens (prompt cố định + 2 tên công ty)
Output: "Apple Inc. and Tesla Inc. are most likely in a competitor relationship."
         → ~20 tokens
Total per call: ~55 tokens
```

#### Step 2 – FactorTemplate
```
Input:  "Please extract the top 5 factors that may affect the stock price of
         {ticker} from the following news: {news_concat}"
         → ~30 tokens (instruction) + ~300 tokens (news text) = ~330 tokens
         [StockNet tweets ~150-200 tokens/ngày, EDT articles ~500-800 tokens]
Output: 5 factors × ~20 tokens = ~100 tokens
Total per call: ~430 tokens (StockNet/CMIN) | ~700 tokens (EDT)
```

#### Step 3 – PriceTemplate
```
Input:  instruction (~35 tokens)
        + factors (~100 tokens)
        + relation (~20 tokens, có thể rỗng)
        + TimeTemplate 5 ngày × ~20 tokens = ~100 tokens
        + final question (~20 tokens)
        = ~275 tokens
Output: prediction + reasoning: ~150-250 tokens
Total per call: ~425-525 tokens (trung bình ~475 tokens)
```

### 7.3 Bảng tính chi tiết theo dataset

**Ký hiệu:**
- `N_test` = số mẫu test
- `α` = tỷ lệ mẫu có stock_match (~0.6)
- Step-1 calls = `N_test × α × 1 match`
- Step-2 calls = `N_test × 1`
- Step-3 calls = `N_test × 1`
- **Total calls** = `N_test × (1 + α + 1)` = `N_test × ~2.6`

---

#### Dataset 1: StockNet (87 stocks, US, tweets)

| Mục | Tính toán | Kết quả |
|---|---|---|
| Mẫu test | ~3,300 | **3,300** |
| Step-1 calls | 3,300 × 0.6 | **~1,980** |
| Step-2 calls | 3,300 × 1 | **3,300** |
| Step-3 calls | 3,300 × 1 | **3,300** |
| **Tổng calls** | 1,980 + 3,300 + 3,300 | **~8,580** |
| Tokens Step-1 | 1,980 × 55 | ~108,900 |
| Tokens Step-2 | 3,300 × 430 | ~1,419,000 |
| Tokens Step-3 | 3,300 × 475 | ~1,567,500 |
| **Tổng tokens** | | **~3,095,400 (~3.1M)** |
| Phần ablation (×3 runs) | 3× | **~9.3M tokens, ~25,740 calls** |

---

#### Dataset 2: CMIN-US (110 stocks, US, tweets)

| Mục | Tính toán | Kết quả |
|---|---|---|
| Mẫu test | ~20,900 | **20,900** |
| Step-1 calls | 20,900 × 0.6 | **~12,540** |
| Step-2 calls | 20,900 × 1 | **20,900** |
| Step-3 calls | 20,900 × 1 | **20,900** |
| **Tổng calls** | | **~54,340** |
| Tokens Step-1 | 12,540 × 55 | ~689,700 |
| Tokens Step-2 | 20,900 × 430 | ~8,987,000 |
| Tokens Step-3 | 20,900 × 475 | ~9,927,500 |
| **Tổng tokens** | | **~19,604,200 (~19.6M)** |
| Phần ablation (×3 runs) | 3× | **~58.8M tokens, ~163,020 calls** |

---

#### Dataset 3: CMIN-CN (300 stocks, CN, tweets)

| Mục | Tính toán | Kết quả |
|---|---|---|
| Mẫu test | ~49,700 | **49,700** |
| Step-1 calls | 49,700 × 0.6 | **~29,820** |
| Step-2 calls | 49,700 × 1 | **49,700** |
| Step-3 calls | 49,700 × 1 | **49,700** |
| **Tổng calls** | | **~129,220** |
| Tokens Step-2 + Step-3 (CN tweets, giả sử ~350 tokens) | | |
| Tokens Step-1 | 29,820 × 55 | ~1,640,100 |
| Tokens Step-2 | 49,700 × 380 | ~18,886,000 |
| Tokens Step-3 | 49,700 × 475 | ~23,607,500 |
| **Tổng tokens** | | **~44,133,600 (~44.1M)** |
| Phần ablation (×3 runs) | 3× | **~132.4M tokens, ~387,660 calls** |

---

#### Dataset 4: EDT (4,228 stocks, US, news articles — **không dùng TimeTemplate**)

| Mục | Tính toán | Kết quả |
|---|---|---|
| Mẫu test | ~14,700 | **14,700** |
| Step-1 calls | 14,700 × 0.6 | **~8,820** |
| Step-2 calls | 14,700 × 1 | **14,700** |
| Step-3 calls (no TimeTemplate) | 14,700 × 1 | **14,700** |
| **Tổng calls** | | **~38,220** |
| Tokens Step-1 | 8,820 × 55 | ~485,100 |
| Tokens Step-2 | 14,700 × 700 (news ~600 tokens dài hơn tweet) | ~10,290,000 |
| Tokens Step-3 (no TimeTemplate, ~350 tokens) | 14,700 × 350 | ~5,145,000 |
| **Tổng tokens** | | **~15,920,100 (~15.9M)** |
| Phần ablation (×3 runs) | 3× | **~47.8M tokens, ~114,660 calls** |

---

### 7.4 Tổng hợp tất cả dataset

| Dataset | Calls (1 run) | Tokens (1 run) | Calls (ablation ×3) | Tokens (ablation ×3) |
|---|---|---|---|---|
| StockNet | **8,580** | **3.1M** | 25,740 | 9.3M |
| CMIN-US | **54,340** | **19.6M** | 163,020 | 58.8M |
| CMIN-CN | **129,220** | **44.1M** | 387,660 | 132.4M |
| EDT | **38,220** | **15.9M** | 114,660 | 47.8M |
| **TỔNG (1 run)** | **~230,360** | **~82.7M** | | |
| **TỔNG (ablation)** | | | **~691,080** | **~248.2M** |

> **Lưu ý:** Các con số trên áp dụng cho **1 LLM model** (GPT-3.5, GPT-4, hoặc GPT-4-turbo). Paper chạy **3 models**, nhân thêm ×3.

### 7.5 Chi phí API ước tính theo từng provider

#### Nếu dùng GPT-3.5-turbo ($0.001/1K input + $0.002/1K output tokens)

| Dataset | Input tokens | Output tokens | Chi phí |
|---|---|---|---|
| StockNet (1 run) | ~2.5M | ~0.6M | **~$3.7** |
| CMIN-US (1 run) | ~15.5M | ~4.1M | **~$23.7** |
| CMIN-CN (1 run) | ~34.6M | ~9.5M | **~$53.6** |
| EDT (1 run) | ~12.8M | ~3.1M | **~$18.8** |
| **Tổng 4 datasets** | | | **~$99.8** |

#### Nếu dùng GPT-4-turbo ($0.01/1K input + $0.03/1K output)

| Dataset | Chi phí (1 run) |
|---|---|
| StockNet | **~$43** |
| CMIN-US | **~$277** |
| CMIN-CN | **~$630** |
| EDT | **~$222** |
| **Tổng 4 datasets** | **~$1,172** |

#### Nếu dùng Gemini 2.0 Flash (miễn phí: 1M tokens/ngày, 15 req/phút)

| Dataset | Tokens cần | Thời gian chạy (15 req/phút) | Số ngày cần |
|---|---|---|---|
| StockNet | 3.1M | 8,580 calls ÷ 15 = 572 phút ≈ 10h | **2-3 ngày** |
| CMIN-US | 19.6M | 54,340 calls ÷ 15 = 3,623 phút ≈ 60h | **12-15 ngày** |
| CMIN-CN | 44.1M | 129,220 calls ÷ 15 = 8,615 phút ≈ 144h | **25-30 ngày** |
| EDT | 15.9M | 38,220 calls ÷ 15 = 2,548 phút ≈ 43h | **10-12 ngày** |

> **Thực tế:** Dùng Gemini free tier thì **CMIN-CN không khả thi** trong thời gian đồ án. Nên chọn **StockNet + CMIN-US** làm 2 dataset chính.

### 7.6 Khuyến nghị dataset cho đồ án (bám sát paper)

| Lựa chọn | Dataset | Chi phí | Thời gian | Ghi chú |
|---|---|---|---|---|
| **Ưu tiên 1** | StockNet | ~$3.7 (GPT-3.5) / miễn phí (Gemini) | 2-3 ngày | Dataset chuẩn nhất, nhỏ nhất |
| **Ưu tiên 2** | CMIN-US | ~$23.7 (GPT-3.5) / ~15 ngày (Gemini) | 2 tuần | Nhiều dữ liệu nhất US |
| **Tùy chọn** | EDT | ~$18.8 (GPT-3.5) / 10-12 ngày (Gemini) | 1.5 tuần | Không cần TimeTemplate |
| **Bỏ qua** | CMIN-CN | ~$630 (GPT-4) / 30 ngày (Gemini) | Quá lớn | Model yếu hơn với CN text |

---

## 8. Metrics Đánh Giá (theo paper)

```python
from sklearn.metrics import accuracy_score, matthews_corrcoef

# ACC = (tp + tn) / (tp + tn + fp + fn)
acc = accuracy_score(y_true, y_pred)

# MCC = (tp×tn - fp×fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))
mcc = matthews_corrcoef(y_true, y_pred)
```

### Kết quả paper cần đạt (Table 2)

| Dataset | Phương pháp | ACC | MCC |
|---|---|---|---|
| **StockNet** | LLMFactor_GPT-4 | **66.32** | **0.238** |
| **CMIN-US** | LLMFactor_GPT-3.5 | **66.42** | **0.288** |
| **CMIN-CN** | LLMFactor_GPT-4-turbo | **60.59** | **0.245** |
| **EDT** | LLMFactor_GPT-4 | **60.83** | **0.105** |

### Kết quả ablation study (Table 3, GPT-3.5-turbo average)

| Layer | StockNet ACC | StockNet MCC |
|---|---|---|
| Price only | 52.16 | 0.041 |
| Price + Factor | 58.04 | 0.166 |
| Price + Factor + Relation | 63.24 | 0.203 |

---

## 9. Độ Khả Thi

| Vấn đề | Mức độ | Giải pháp |
|---|---|---|
| **API rate limit (Gemini free: 15 req/phút)** | CAO | Thêm `time.sleep(4)` giữa các calls; cache pkl |
| **Reproducibility (LLM non-deterministic)** | TRUNG BÌNH | Cố định `temperature=0`; ghi log raw response |
| **Thời gian chạy CMIN-CN quá lớn** | CAO | Chỉ chạy StockNet + CMIN-US; bỏ CMIN-CN |
| **Không đủ VRAM cho FinGPT-13B baseline** | TRUNG BÌNH | Dùng Colab Pro hoặc bỏ FinGPT baseline |
| **Matching news → stock_list không chính xác** | TRUNG BÌNH | Dùng string matching + fuzzy match |

---

## 10. Cấu Trúc Source Code (bám sát paper)

```
SKGP/
├── docs/
│   ├── 2406.10811v1.pdf        # Paper gốc
│   └── tasks.md                # File này
├── data/
│   ├── stocknet/               # StockNet raw data
│   │   ├── price/              # {TICKER}.csv
│   │   └── tweet/              # {TICKER}/{DATE} JSON files
│   ├── cmin_us/                # CMIN-US raw data
│   ├── cmin_cn/                # CMIN-CN raw data
│   └── edt/                    # EDT raw data
├── src/
│   ├── data_loader.py          # Load + preprocess + split
│   ├── skgp_pipeline.py        # 3-step SKGP main logic
│   ├── baselines/
│   │   ├── keyphrase.py        # PromptRank, KeyBERT, YAKE, TextRank, TFIDF
│   │   └── sentiment.py        # FinBERT, RoBERTa, FinGPT, GPT sentiment
│   └── evaluate.py             # ACC, MCC, confusion matrix
├── cache/                      # Cache LLM responses (pkl)
│   ├── stocknet_step1.pkl
│   ├── stocknet_step2.pkl
│   └── stocknet_step3.pkl
├── results/
│   └── results_table.csv       # Bảng kết quả đầy đủ
├── main.py                     # Entry point: chạy SKGP pipeline
└── requirements.txt
```

---

## 11. Tài Liệu Tham Khảo

| Tài liệu | Link |
|---|---|
| Paper LLMFactor | https://arxiv.org/abs/2406.10811 |
| StockNet dataset | https://github.com/yumoxu/stocknet-dataset |
| CMIN dataset | https://github.com/wuhuizhe/CHRNN |
| EDT dataset | https://github.com/Zhihan1996/TradeTheEvent |
| FinBERT (EN) | https://huggingface.co/ProsusAI/finbert |
| Financial RoBERTa | https://huggingface.co/soleimanian/financial-roberta-large-sentiment |
| FinGPT | https://huggingface.co/FinGPT/fingpt-sentiment_llama2-13b_lora |
| pke toolkit | https://github.com/boudinfl/pke |
| Gemini API | https://ai.google.dev/gemini-api/docs |
