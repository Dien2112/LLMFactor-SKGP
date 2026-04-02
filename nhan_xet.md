# Nhan xet: Code vs Paper (arXiv:2406.10811v1)

**Paper**: "LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction"  
**Authors**: Meiyun Wang, Kiyoshi Izumi, Hiroki Sakaji  
**Ngay review**: 2026-04-02

---

## 1. Tong quan

Code implement framework LLMFactor voi pipeline SKGP (Sequential Knowledge-Guided Prompting) gom 3 buoc:
- Step 1: Relation Extraction (Background Knowledge)
- Step 2: Factor Extraction
- Step 3: Prediction

**Nhan xet chung**: Pipeline 3 buoc duoc implement **dung logic** cua paper. Cac template prompt co ban sat voi Appendix A (Table 5) cua paper. Tuy nhien co mot so diem khac biet can luu y.

---

## 2. So sanh chi tiet tung buoc

### 2.1 Step 1 - Relation Extraction

| Tieu chi | Paper (Section 3.2.1, Table 5) | Code (`skgp.py:248-273`) |
|---|---|---|
| RelationTemplate | "Please fill in the blank and return a complete sentence: stock_target and stock_match are most likely in a ___ relationship." | "Please fill in the blank and return a complete sentence: {target_name} and {match_name} are most likely in a ___ relationship." |
| Matching | S = {(C_i, T_i, I_i)} match voi news_target | Tim cashtag ($TICKER) va company name trong tweets |
| Cache key | Khong de cap | (target_ticker, match_ticker) - **hop ly**, relation khong phu thuoc date |

**Dung**:
- Template prompt khop voi Appendix A Table 5 cua paper.
- Fill-in-the-blank strategy dung theo paper.
- Dung company name (vd: "Apple Inc.") thay vi ticker trong prompt - hop ly hon vi LLM hieu company name tot hon.

**Luu y**:
- `MAX_RELATIONS = 3`: Paper **khong** gioi han so luong stock_match. Day la tham so tu them, co the anh huong ket qua neu nhieu hon 3 related companies.
- Stock matching (`find_stock_matches`): Logic kiem tra cashtag/company name trong tweets la hop ly, tuy nhien paper mo ta formal hon bang phep giao tap hop S ∩ news_target.

---

### 2.2 Step 2 - Factor Extraction

| Tieu chi | Paper (Section 3.2.2, Table 5) | Code (`skgp.py:280-306`) |
|---|---|---|
| FactorTemplate | "Please extract the top k factors that may affect the stock price of stock_target from the following news." + news_target | "Please extract the top {K_FACTORS} factors that may affect the stock price of {company} ({ticker}) from the following news:\n{news_text}" |
| k (so factors) | k = 5 (Section 4.4) | K_FACTORS = 5 |

**Dung**:
- Template **khop chinh xac** voi paper.
- K_FACTORS = 5 dung voi paper (Section 4.4: "the number of keyphrases and factors k is 5").

**Luu y**:
- `MAX_TWEETS = 10`: Paper **khong** gioi han so tweet trong prompt. Tham so nay tu them, co the mat thong tin neu ngay do co nhieu hon 10 tweets.
- Tweets duoc noi bang " | " - cach join nay hop ly nhung paper khong specify cu the format noi tweets.

---

### 2.3 Step 3 - Prediction

| Tieu chi | Paper (Section 3.2.3, Table 5 Appendix A) | Code (`skgp.py:325-369`) |
|---|---|---|
| PriceTemplate mo dau | "Based on the following information, please judge the direction of the stock price **from** rise/fall, fill in the blank and give reasons." | "Based on the following information, please judge the direction of the stock price **as** rise or fall, fill in the blank and give reasons." |
| Factor section | "These are the main factors that may affect this stock's price recently: factor." | "These are the main factors that may affect this stock's price recently:\n{factors}" |
| Relation section | "These are the connections between the companies that have appeared in the news: relation." | "These are the connections between the companies that have appeared in the news:\n{relation_str}" |
| TimeTemplate | "On date_i, the stock price of stock_target f(P_i)." (rose/fell) | "On {date}, the stock price of {company} ({ticker}) rose/fell." |
| Fill-in-the-blank cuoi | "On date_i, the stock price of stock_target will ___." | "On {date}, the stock price of {company} ({ticker}) will ___." |

**Dung**:
- Tat ca cac thanh phan cua prompt Step 3 **khop voi paper** (Appendix A, Table 5).
- TimeTemplate dung: chuyen price history thanh text format "rose"/"fell".
- Fill-in-the-blank cuoi cung dung.

**Khac voi paper**:
- Code dung `use_strong_model=True` cho Step 3 (model manh hon cho prediction). Tuy nhien, **paper dung cung 1 model cho ca 3 buoc** (VD: LLMFactor_GPT-4 = GPT-4 cho Step 1, 2, va 3). Viec tach standard/strong model pool la thiet ke rieng cua code, **khong co trong paper**.

**Luu y nho**:
- "as rise or fall" vs "from rise/fall" - khac biet nho ve wording, khong anh huong nghi.
- Prediction parse: `pred = 1 if "rise" in raw.lower() else 0` - cach nay don gian nhung co the sai neu LLM tra loi kieu "the stock will not rise" -> van detect "rise". Paper khong de cap cach parse cu the nhung day la diem can can than.

---

### 2.4 Hyperparameters

| Tham so | Paper (Section 4.4) | Code | Khop? |
|---|---|---|---|
| Window size (t) | 5 | WINDOW_SIZE = 5 | **Dung** |
| Top-k factors | 5 | K_FACTORS = 5 | **Dung** |
| MAX_TWEETS | Khong de cap | 10 | Tu them |
| MAX_RELATIONS | Khong de cap | 3 | Tu them |
| Temperature | Khong de cap | 0.0 | Hop ly cho reproducibility |
| max_output_tokens | Khong de cap | 512 | Hop ly |

---

### 2.5 Evaluation Metrics

| Metric | Paper (Section 4.2) | Code (`skgp.py:535-563`) | Khop? |
|---|---|---|---|
| ACC | (tp+tn)/(tp+fp+fn+tn) | sklearn.metrics.accuracy_score | **Dung** |
| MCC | Formula (2) trong paper | sklearn.metrics.matthews_corrcoef | **Dung** |
| Confusion Matrix | Co | sklearn.metrics.confusion_matrix | **Dung** |

---

### 2.6 Data Format & Label

| Tieu chi | Paper | Code | Khop? |
|---|---|---|---|
| Label | Rise (1) neu gia tang, Fall (0) neu giam | `(ret_close > 0).astype(int)` | **Dung** |
| Dataset | StockNet (2014-2016, 87 stocks) | Load tu StockNet format | **Dung** |
| Test period | Khong ro trong paper | 2015-10-01 den 2016-01-01 | Hop ly |

---

## 3. Diem khac biet QUAN TRONG

### 3.1 Model LLM khac voi paper (CRITICAL)

| | Paper | Code |
|---|---|---|
| Cach dung model | **Cung 1 model cho ca 3 buoc** (VD: LLMFactor_GPT-4 = GPT-4 cho Step 1, 2, 3) | **Tach 2 pool**: standard (Step 1-2) va strong (Step 3) |
| Models | GPT-3.5-turbo, GPT-4, GPT-4-turbo | Standard: gemma-3-1b/4b/12b-it, gemini-2.5-flash. Strong: gemma-3-27b-it, gemini-2.5-pro |

**Van de**:
- **Paper dung cung 1 model cho ca 3 buoc**. Code tach thanh 2 pool (standard/strong) la thiet ke rieng, **khong co trong paper**. Dieu nay lam thay doi behavior cua pipeline va kho so sanh ket qua voi paper.
- Paper bao cao ket qua voi GPT-3.5-turbo, GPT-4, GPT-4-turbo. Code dung Gemini/Gemma models hoan toan khac.
- **gemma-3-1b-it va gemma-3-4b-it la model rat nho** (1B va 4B params). Kha nang extract factor va relation se kem hon nhieu so voi GPT-3.5-turbo (model yeu nhat cua paper). Dieu nay se **anh huong truc tiep den chat luong ket qua**.
- Round-robin qua nhieu model khac nhau (1B -> 4B -> 12B -> Flash) co nghia la **cung mot pipeline co the cho ket qua khac nhau** tuy vao model nao dang active tai thoi diem goi, giam reproducibility.

### 3.2 Industry/Sector khong duoc su dung trong prompt

Paper dinh nghia stock list S = {(C_i, T_i, **I_i**)} bao gom industry. Code load sector/industry tu StockTable nhung **khong dua vao prompt nao**. Tuy nhien, paper cung khong explicitly su dung industry trong template, nen day khong phai loi.

### 3.3 main.py chua hoat dong

```python
from SKGP import skgp;  # import sai ten file (SKGP vs skgp)
def main():
    print("Main")
    #skgp();  # commented out
```

- File `main.py` import sai (`SKGP` vs `skgp` - case-sensitive tren Linux).
- Function `skgp()` bi comment out, main.py khong lam gi.

---

## 4. Van de ky thuat trong code

### 4.1 Prediction parsing co the sai

```python
# skgp.py:368
pred = 1 if "rise" in raw.lower() else 0
```

Neu LLM tra loi "the stock will **not rise**" hoac "there is **no reason to expect a rise**", code van detect "rise" va predict = 1. Can phan tich context hon hoac dung regex chinh xac hon (vd: kiem tra "will rise" vs "will fall").

### 4.2 Tao llm instance moi moi lan goi skgp()

```python
# skgp.py:404
def skgp(...):
    llm_instance = llm()  # tao moi moi lan goi
```

Moi lan goi `skgp()`, mot `GeminiLLMManager` moi duoc tao, reset usage stats va pool index. Trong `test.py`, goi `skgp()` 8 lan = tao 8 manager instances. Nen truyen llm instance tu ben ngoai hoac dung singleton.

### 4.3 Cache load/save moi lan goi skgp()

```python
# skgp.py:413-415
cache1 = load_cache(cache1_path)
cache2 = load_cache(cache2_path)
cache3 = load_cache(cache3_path)
```

Moi lan goi `skgp()` deu doc 3 file pickle va ghi lai 3 file pickle. Voi backtest nhieu samples, dieu nay khong hieu qua. Function `run_pipeline()` (line 448) xu ly tot hon - load cache 1 lan roi reuse.

### 4.4 Naming conflict: llm class vs parameter

```python
# skgp.py:248
def step1_get_relation(llm: llm, ...):  # parameter "llm" shadows class "llm"
```

Trong `step1_get_relation`, `step2_get_factors`, `step3_get_prediction`: parameter name `llm` trung voi class name `llm`. Khong gay loi nhung confusing.

### 4.5 run_pipeline() tham so llm type hint sai

```python
# skgp.py:450
def run_pipeline(samples, llm: GeminiLLM, ...):  # GeminiLLM chua duoc define/import
```

Type hint `GeminiLLM` khong ton tai (class thuc te la `GeminiLLMManager`). Function nay cung khong duoc goi tu dau trong codebase hien tai.

---

## 5. Tom tat

### Dung voi paper:
- Pipeline SKGP 3 buoc (Relation -> Factor -> Prediction): **DUNG**
- Tat ca prompt templates: **KHOP** voi Appendix A, Table 5
- Hyperparameters (window_size=5, k=5): **DUNG**
- Evaluation metrics (ACC, MCC): **DUNG**
- Fill-in-the-blank strategy: **DUNG**
- TimeTemplate format: **DUNG**
- Data loading & label: **DUNG** (StockNet format, ret_close > 0 = rise)
- Caching strategy (relation theo ticker pair, factor/predict theo ticker+date): **HOP LY**

### Khac voi paper:
- Model LLM: **KHAC** (Gemini/Gemma thay vi GPT family) - **anh huong lon den ket qua**
- **Dual model pool (standard/strong)**: Paper dung **cung 1 model cho ca 3 buoc**, code tach 2 pool - **KHONG CO** trong paper
- Model pool round-robin: **KHONG CO** trong paper - giam reproducibility
- MAX_TWEETS=10, MAX_RELATIONS=3: **TU THEM** - paper khong gioi han
- Prediction parsing don gian (`"rise" in response`): co the **SAI** voi phu dinh

### Loi ky thuat:
- `main.py` import sai case (`SKGP` vs `skgp`) va commented out
- Tao llm instance moi moi lan goi `skgp()` (khong hieu qua)
- Cache load/save moi sample (khong hieu qua khi batch)
- Type hint `GeminiLLM` khong ton tai
- Parameter name `llm` shadow class name `llm`

### Danh gia tong the:

| Hang muc | Diem (1-10) | Ghi chu |
|---|---|---|
| Logic pipeline | 9/10 | 3 buoc SKGP dung paper |
| Prompt templates | 9/10 | Gan nhu giong het Appendix A |
| Hyperparameters | 10/10 | Khop hoan toan |
| Evaluation | 10/10 | ACC + MCC dung paper |
| Model selection | 4/10 | Khac paper hoan toan, model nho co the cho ket qua kem |
| Code quality | 6/10 | Co cac van de ky thuat nhu tren |
| Completeness | 7/10 | main.py chua hoat dong, run_pipeline co type hint sai |

**Ket luan**: Code implement **dung ve mat thuat toan va prompt design** cua paper LLMFactor. Diem yeu chinh la su dung model LLM khac (Gemini/Gemma thay vi GPT) voi cac model nho (1B, 4B) co the lam giam chat luong ket qua dang ke so voi paper goc. Ngoai ra co mot so loi ky thuat nho can sua.
