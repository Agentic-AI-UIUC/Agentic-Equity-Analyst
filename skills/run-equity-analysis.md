---
name: run-equity-analysis
description: Run a full equity research analysis for a company. Use when the user asks to analyze, research, or generate a report for a stock or company.
---

# Run Equity Analysis

Run a full equity research analysis for a company using the agentic pipeline.

## Steps

1. Ensure the virtual environment is active and `.env` is populated.
2. Ask the user for the company name, ticker, and year if not already provided.
3. Run the analysis:

```bash
source .venv/bin/activate
python main.py --company "$COMPANY" --ticker $TICKER --year $YEAR
```

To also launch the Streamlit UI after the report is generated:

```bash
python main.py --company "$COMPANY" --ticker $TICKER --year $YEAR --launch-ui
```

4. The report will be written to `report.txt`. Share key findings with the user.

## Notes
- Ensure `filing_embedder.py` and `news_loader.py` have been run for the ticker first (see `/hydrate-vectors`).
- If `gpt-5.1` is unavailable, downgrade `LLM_MANAGER` to `gpt-4o` in `reporting_pipeline.py`.
