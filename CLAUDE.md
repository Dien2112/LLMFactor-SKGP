# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the LLMFactor SKGP (Sequential Knowledge-Guided Prompting) framework for stock movement prediction, based on arXiv paper 2406.10811v1. Uses Google Gemini LLM API to predict whether stock prices will rise or fall by analyzing tweets, inter-company relationships, and historical price data.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (backtest against 8 fixed fixtures in test_data/)
python test.py

# Run main pipeline (currently stub - skgp() call is commented out)
python main.py
```

## Environment Setup

Requires a `.env` file in project root with:
```
GEMINI_API_KEY=<your_key>
```

## Architecture

### 3-Step Pipeline (`skgp.py`)

The core `skgp()` function runs a sequential pipeline for each (ticker, date) pair:

1. **Step 1 - Relation Extraction** (`step1_get_relation`): Finds other stocks mentioned in tweets via cashtags/company names, then asks LLM to describe the relationship between companies. Uses standard models. Cache key: `(target_ticker, match_ticker)` (date-independent).

2. **Step 2 - Factor Extraction** (`step2_get_factors`): Extracts top-K price impact factors from tweets using LLM. Uses standard models. Cache key: `(ticker, date)`.

3. **Step 3 - Prediction** (`step3_get_prediction`): Combines factors, relations, and price history into a prompt, asks LLM to predict rise/fall. Uses **strong models** (`use_strong_model=True`). Cache key: `(ticker, date)`. Prediction parsed by checking if "rise" appears in response.

### LLM Layer (`llm/`)

- `llm/llm.py` - Thin wrapper (`class llm`) that delegates to `GeminiLLMManager`. Swapping LLM providers means modifying only this file.
- `llm/geminillm.py` - Manages two model pools (standard and strong). Implements rate-limiting (per-model RPM intervals) and round-robin rotation on 429/409 errors. Retries up to `num_models * 2` attempts before raising `RuntimeError`.

**Standard models** (Steps 1 & 2): gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it, gemini-2.5-flash
**Strong models** (Step 3): gemma-3-27b-it, gemini-2.5-pro

### Caching

Each pipeline step uses separate pickle caches (`step1_relation.pkl`, `step2_factor.pkl`, `step3_predict.pkl`). Caches are saved atomically via temp file + rename. This enables resumable batch runs - previously processed samples skip API calls.

### Data Format

- **Price data**: Tab-separated `.txt` files in `data/price/preprocessed/`. Columns: date, ret_open, ret_high, ret_low, ret_close, price_change, volume. Label derived from `ret_close > 0`.
- **Tweet data**: JSON-lines files in `data/tweet/preprocessed/{TICKER}/{DATE}`. Each line: `{"text": ["token1", "token2", ...]}`.
- **StockTable**: Tab-separated file at `data/StockTable` mapping `Sector\t$TICKER\tCompany`.
- **Test fixtures**: JSON files in `test_data/` with keys: stock_target, date, label, tweets, history.

### Key Hyperparameters (paper Section 4.4)

```python
WINDOW_SIZE   = 5    # historical price sessions in prompt
K_FACTORS     = 5    # top-k factors to extract
MAX_TWEETS    = 10   # max tweets per prompt
MAX_RELATIONS = 3    # max related stocks per sample
```

### Evaluation

`evaluate()` computes accuracy (ACC) and Matthews Correlation Coefficient (MCC) using scikit-learn. Benchmark target: LLMFactor-GPT-4 achieves ACC=66.32%, MCC=0.238.
