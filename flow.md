# LLMFactor - Sequential Knowledge-Guided Prompting (SKGP) Flow

> Paper: "LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction" (arXiv:2406.10811v1)

## Overview

LLMFactor su dung chien luoc **Sequential Knowledge-Guided Prompting (SKGP)** de du doan xu huong tang/giam cua co phieu. Pipeline gom 3 buoc tuan tu, moi buoc tao ra knowledge de dan dat buoc tiep theo.

**Input:** `stock_target`, `date_target`, `news_target` (tweets/tin tuc), `P` (lich su gia WINDOW_SIZE=5 phien)
**Output:** Du doan `rise` hoac `fall` kem ly do giai thich

---

## Step 0 - Data Preparation (Chuan bi du lieu)

1. Load **StockTable** - bang anh xa `(Sector, $TICKER, Company)` tu NYSE va NASDAQ
2. Load **Price data** - chuoi gia lich su, chuyen thanh chuoi movement `P_hat` (1=rise, 0=fall) dua tren `ret_close > 0`
3. Load **Tweet/News data** - tin tuc lien quan den co phieu tai `date_target`
4. Xay dung tap test samples: moi sample gom `(ticker, date, tweets, history[5 phien], label)`

---

## Step 1 - Relation Extraction (Trich xuat quan he giua cac cong ty)

**Muc dich:** Tao background knowledge ve moi quan he giua `stock_target` va cac cong ty duoc nhac den trong tin tuc.

### 1a. Matching (Tim cong ty lien quan)
- Quet toan bo tweets de tim cac **cashtag** (`$TICKER`) hoac **ten cong ty** xuat hien
- Doi chieu voi StockTable de xac dinh `stock_match` (loai bo chinh `stock_target`)
- Gioi han toi da **MAX_RELATIONS = 3** cong ty

### 1b. Relation Prompting (Hoi LLM ve quan he)
- Voi moi cap `(stock_target, stock_match)`, gui prompt theo **RelationTemplate**:

```
"Please fill in the blank and return a complete sentence:
{target_company} and {match_company} are most likely in a ___ relationship."
```

- LLM tra ve mieu ta quan he (vd: competitive, supply-chain, partnership...)
- **Cache key:** `(target_ticker, match_ticker)` - khong phu thuoc ngay (quan he giua 2 cong ty la co dinh)
- **Model:** Standard models (gemma-3-1b/4b/12b-it, gemini-2.5-flash)

**Output:** List cac `relation` string

---

## Step 2 - Factor Extraction (Trich xuat cac yeu to anh huong gia co phieu)

**Muc dich:** Tu tin tuc, trich xuat top-K yeu to co the anh huong den gia co phieu.

### Prompt theo FactorTemplate:

```
"Please extract the top {K_FACTORS=5} factors that may affect the stock price
of {company} ({ticker}) from the following news:
{news_text}"    (toi da MAX_TWEETS=10 tweets, noi bang " | ")
```

- LLM phan tich noi dung tin tuc va xac dinh cac yeu to anh huong
- Factors khong gioi han trong tu trong tin tuc - LLM co the tong hop va suy luan
- Vd: "Nvidia stock gain in January", "new product announcements", "selection of Nvidia Drive Thor by EV makers"
- **Cache key:** `(ticker, date)`
- **Model:** Standard models

**Output:** `factors` string (danh sach cac yeu to)

---

## Step 3 - Prediction (Du doan xu huong gia)

**Muc dich:** Ket hop tat ca thong tin tu Step 1, Step 2 va lich su gia de du doan rise/fall.

### 3a. Xay dung TimeTemplate (Chuyen lich su gia thanh text)
- Voi moi phien trong `history` (WINDOW_SIZE=5 phien gan nhat):

```
"On {date}, the stock price of {company} ({ticker}) rose/fell."
```

### 3b. Xay dung PriceTemplate va gui prompt tong hop:

```
"Based on the following information, please judge the direction of the
stock price as rise or fall, fill in the blank and give reasons.

These are the main factors that may affect this stock's price recently:
{factors}                          <-- tu Step 2

These are the connections between the companies that have appeared in the news:
{relations}                        <-- tu Step 1

On {date1}, the stock price of {company} rose.
On {date2}, the stock price of {company} fell.
...                                <-- TimeTemplate (5 phien)

On {date_target}, the stock price of {company} ({ticker}) will ___."
```

- LLM dien vao cho trong va dua ra ly do
- **Parse ket qua:** Kiem tra neu "rise" xuat hien trong response -> pred=1 (rise), nguoc lai -> pred=0 (fall)
- **Cache key:** `(ticker, date)`
- **Model:** Strong models (gemma-3-27b-it, gemini-2.5-pro) - dung model manh hon vi day la buoc quan trong nhat

**Output:** `prediction` (0 hoac 1) + `raw_response` (giai thich)

---

## Step 4 - Evaluation (Danh gia ket qua)

- **Accuracy (ACC):** Ti le du doan dung
- **Matthews Correlation Coefficient (MCC):** Do tuong quan giua du doan va thuc te (xu ly tot truong hop mat can bang du lieu)
- **Confusion Matrix:** TP, TN, FP, FN
- **Benchmark:** LLMFactor_GPT-4 dat ACC=66.32%, MCC=0.238 tren StockNet dataset

---

## Flow Diagram

```
Input: stock_target, date_target, tweets, price_history
                        |
                        v
        +-------------------------------+
        |  Step 0: Data Preparation     |
        |  - Load StockTable            |
        |  - Load Price + Tweets        |
        |  - Build test samples         |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  Step 1: Relation Extraction  |
        |  1a. Match stock in tweets    |
        |  1b. LLM: fill-in-the-blank  |
        |      -> relations             |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  Step 2: Factor Extraction    |
        |  LLM: extract top-K factors   |
        |  from news/tweets             |
        |      -> factors               |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  Step 3: Prediction           |
        |  3a. Build TimeTemplate       |
        |  3b. Combine all info:        |
        |      relations + factors +    |
        |      TimeTemplate             |
        |  3c. LLM (strong model):      |
        |      predict rise/fall        |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  Step 4: Evaluation           |
        |  ACC + MCC + Confusion Matrix |
        +-------------------------------+
```

---

## Ablation Study (Dong gop cua tung thanh phan)

| Layer              | ACC contribution | MCC contribution |
|--------------------|-----------------|-----------------|
| Price only (TimeTemplate) | ~86% | ~32% |
| + Factor (Step 2)  | +9%             | +46%            |
| + Relation (Step 1)| +5%             | +22%            |

-> **Factor layer dong gop nhieu nhat** cho hieu suat tong the cua LLMFactor.

---

## Hyperparameters (Section 4.4)

| Parameter       | Value | Mo ta                                    |
|----------------|-------|------------------------------------------|
| WINDOW_SIZE    | 5     | So phien gia lich su trong prompt         |
| K_FACTORS      | 5     | So luong factors trich xuat              |
| MAX_TWEETS     | 10    | So tweet toi da trong prompt             |
| MAX_RELATIONS  | 3     | So cong ty lien quan toi da              |

---

## Diem noi bat cua SKGP

1. **Fill-in-the-blank technique:** Gioi han format response, giup LLM tra loi truc tiep va ro rang
2. **Sequential prompting:** Moi buoc tao knowledge cho buoc sau, giup LLM co ngon context hon
3. **Factor > Keyphrase/Sentiment:** Factors lien quan truc tiep den gia co phieu, co kha nang giai thich va doc hieu tot hon
4. **Text-formatted time-series:** Chuyen du lieu gia thanh van ban de LLM co the hieu va xu ly
5. **Explainability:** Moi du doan kem theo ly do cu the, giup nguoi dung hieu logic dang sau
