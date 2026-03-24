"""
competitor_agent_hardcoded.py
─────────────────────────────
Returns the top 5 widely-accepted competitors for any S&P 500 company.

Source: sp500_competitors.json — hand-curated peer groups reflecting
what is broadly accepted on financial sites, analyst reports, and
general public knowledge. Fully deterministic — no API calls, no ML.

Usage
─────
  CLI :  python competitor_agent_hardcoded.py GOOGL
  Tool:  from competitor_agent_hardcoded import competitor_tool_hardcoded
"""

import json
import sys
from pathlib import Path
from typing import List

from langchain.tools import tool

COMPETITORS_FILE = Path(__file__).parent / "sp500_competitors.json"

with open(COMPETITORS_FILE) as _f:
    _DATA: dict = json.load(_f)


# ─── CORE FUNCTION ────────────────────────────────────────────────────────────

def get_competitors(ticker: str) -> List[str]:
    """Return the list of competitor tickers for the given ticker."""
    ticker = ticker.upper().strip()
    if ticker not in _DATA:
        raise KeyError(f"'{ticker}' not found in competitor database.")
    return _DATA[ticker].get("competitors", [])


# ─── LANGCHAIN TOOL ───────────────────────────────────────────────────────────

@tool
def competitor_tool_hardcoded(ticker: str) -> str:
    """
    Returns the 5 most widely-accepted competitors for a given S&P 500 ticker.

    Uses a static JSON database (sp500_competitors.json) that encodes broadly
    accepted peer groups based on web consensus — analyst reports, financial
    sites, and public knowledge.

    Fully deterministic — same ticker always returns the same competitors.
    No API calls required. Covers all ~500 S&P 500 constituents.

    Returns competitor ticker symbols only, one per line.
    Takes one string argument: the stock ticker symbol (e.g., 'AAPL', 'NVDA').
    """
    try:
        return "\n".join(get_competitors(ticker))
    except KeyError as e:
        return f"[ERROR] {e}"


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(get_competitors(t))
