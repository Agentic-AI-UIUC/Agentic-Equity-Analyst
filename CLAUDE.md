# Agentic Equity Analyst — Claude Code Instructions

## Project Overview
LangChain-powered multi-agent equity research system. A **manager agent** (gpt-5.1) delegates to a **reporting agent** (gpt-4o) which calls specialized tools to produce professional equity research writeups.

## Architecture

```
main.py  →  reporting_pipeline.py
              ├── manager_agent  (gpt-5.1, deepagents)
              │     └── create_report tool
              └── reporting_agent (gpt-4o, langchain)
                    ├── analyze_filings      → analyst.py / ChromaDB: company_filings
                    ├── analyze_parser       → analyst.py / ChromaDB: parser_data
                    ├── analyze_financials   → analyst.py / ChromaDB: financial_data
                    ├── analyze_news         → analyst.py / ChromaDB: news_articles
                    ├── find_dcf_tool        → dcf.py / yfinance + ChromaDB: parser_data
                    ├── calculate_moving_average_tool → market_data_loader.py
                    └── valuation_tool       → valuation_agent.py / ChromaDB: valuation_memos
```

## Key Files
| File | Role |
|---|---|
| `main.py` | CLI entry point (`--company`, `--ticker`, `--year`, `--launch-ui`) |
| `reporting_pipeline.py` | Manager + reporting agents, `generate_financial_report()` |
| `analyst.py` | `@tool`-decorated RAG wrappers for 4 Chroma collections |
| `dcf.py` | DCF math + Yahoo Finance data fetching + `find_dcf_tool` |
| `market_data_loader.py` | Moving average tool + yfinance ingestion |
| `valuation_agent.py` | Valuation memo RAG tool |
| `filing_embedder.py` | One-time: embed FMP 10-K/10-Q JSON into Chroma |
| `news_loader.py` | One-time: embed Perplexity Sonar news into Chroma |
| `pdf_builder.py` | Writes `report.txt` to disk, optionally launches Streamlit |
| `streamlit_app.py` | Streamlit UI reading `report.txt` |

## Environment Variables (`.env`)
```
OPENAI_API_KEY       # GPT-4o + text-embedding-3-small
CHROMADB             # Chroma Cloud DB name
CHROMADB_API_KEY     # Chroma Cloud service token
CHROMADB_TENANT      # Chroma Cloud tenant slug
FMP_API_KEY          # Financial Modeling Prep (filings)
PPLX_API_KEY         # Perplexity Sonar (news)
FRED_API_KEY         # Optional: FRED macro data
```

## Chroma Collections
| Collection | Populated by | Used by tool |
|---|---|---|
| `company_filings` | `filing_embedder.py` | `analyze_filings` |
| `parser_data` | `filing_embedder.py` | `analyze_parser`, `find_dcf_tool` |
| `financial_data` | `market_data_loader.py` | `analyze_financials` |
| `news_articles` | `news_loader.py` | `analyze_news` |
| `valuation_memos` | `valuation_agent.py` | `valuation_tool` |

## Running the Agent
```bash
source .venv/bin/activate
python main.py --company "Nvidia" --ticker NVDA --year 2026
python main.py --company "Apple" --ticker AAPL --year 2026 --launch-ui
```

## Hydrating Vector Stores (run before first analysis)
```bash
python filing_embedder.py          # SEC 10-K/10-Q from FMP
python news_loader.py --ticker AAPL --time-range 1m
# market_data_loader is called automatically via the LangChain tool
```

## Adding a New Tool
1. Create a new `.py` file or add to an existing one
2. Decorate the function with `@tool` from `langchain.tools`
3. Add it to the `reporting_tools` list in `reporting_pipeline.py`
4. Re-initialize the `reporting_agent` (automatic on next import)

## Conventions
- All LangChain tools must accept only simple Python types (str, int) — no complex objects
- Chroma connections use Cloud (not local) — always pass `database`, `chroma_cloud_api_key`, `tenant`
- Embedding model: `text-embedding-3-small` (hardcoded in `EMBEDDING_MODEL`)
- Output narrative: written to `report.txt` via `pdf_builder.report()`
- Python venv: `.venv/` — activate before running anything
- LLM models: manager=`gpt-5.1`, reporter=`gpt-4o`

## Common Issues
- **ChromaDB auth error** → check `CHROMADB`, `CHROMADB_API_KEY`, `CHROMADB_TENANT` in `.env`
- **Empty DCF result** → `find_dcf` relies on `parser_data` to extract ticker; ensure `filing_embedder.py` has been run for that company
- **Tool timeout** → `LLM_REPORTER` has `timeout=30`; increase if needed for complex queries
- **`gpt-5.1` unavailable** → downgrade `LLM_MANAGER` to `gpt-4o` in `reporting_pipeline.py`

## Skills
Use these Claude Code skills (trigger with `/`):
- `/run-equity-analysis` — run a full analysis for a company
- `/hydrate-vectors` — populate Chroma vector stores for a ticker
- `/add-equity-tool` — scaffold a new LangChain tool for the reporting agent
