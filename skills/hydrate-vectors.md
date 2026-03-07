---
name: hydrate-vectors
description: Populate ChromaDB vector stores for a ticker before analysis. Use when the user asks to load, embed, or hydrate data for a company or ticker.
---

# Hydrate Vector Stores

Populate ChromaDB vector stores for a given ticker before running analysis.

## Steps

1. Activate the virtual environment:

```bash
source .venv/bin/activate
```

2. Ask the user for the ticker symbol if not provided.

3. Embed SEC filings (10-K / 10-Q) from Financial Modeling Prep:

```bash
python filing_embedder.py
```

4. Embed recent news articles via Perplexity Sonar (default: last 1 month):

```bash
python news_loader.py --ticker $TICKER --time-range 1m
```

Available time ranges: `1d`, `1w`, `1m`, `3m`, `1y`.

5. Market/financial data (`financial_data` collection) is loaded automatically when the `calculate_moving_average_tool` runs during analysis — no manual step needed.

## Collections Populated
| Collection | Script |
|---|---|
| `company_filings` | `filing_embedder.py` |
| `parser_data` | `filing_embedder.py` |
| `news_articles` | `news_loader.py` |
| `financial_data` | auto via `market_data_loader.py` |

## Notes
- Requires `FMP_API_KEY` and `PPLX_API_KEY` in `.env`.
- Run this before the first `/run-equity-analysis` for any new ticker.
