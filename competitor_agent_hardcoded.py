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
from typing import Dict

from langchain.tools import tool

COMPETITORS_FILE = Path(__file__).parent / "sp500_competitors.json"


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def _load() -> Dict:
    with open(COMPETITORS_FILE, "r") as f:
        return json.load(f)


# ─── FORMATTED OUTPUT ─────────────────────────────────────────────────────────

def find_competitors_hardcoded(ticker: str) -> str:
    """Full formatted output for CLI / human use."""
    ticker = ticker.upper().strip()
    data   = _load()

    if ticker not in data:
        return (
            f"[ERROR] '{ticker}' not found in hardcoded competitor database.\n"
            f"        Covered tickers: {len(data)}  |  File: {COMPETITORS_FILE.name}"
        )

    entry       = data[ticker]
    name        = entry.get("name", ticker)
    competitors = entry.get("competitors", [])

    if not competitors:
        return f"[ERROR] No competitors stored for {ticker}."

    W     = 64
    dline = "═" * W
    line  = "─" * W

    lines = [
        dline,
        f"  HARDCODED COMPETITOR ANALYSIS: {ticker}  ({name})",
        f"  Source: sp500_competitors.json  (static, web-consensus peer groups)",
        dline,
        f"  Top {len(competitors)} widely-accepted competitors:",
        line,
    ]

    for i, comp in enumerate(competitors, 1):
        comp_name = data.get(comp, {}).get("name", comp)
        lines.append(f"  {i}.  {comp:<8}  {comp_name}")

    lines.append(dline)
    return "\n".join(lines)


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
    ticker = ticker.upper().strip()
    data   = _load()

    if ticker not in data:
        return f"[ERROR] '{ticker}' not found in hardcoded competitor database."

    competitors = data[ticker].get("competitors", [])
    if not competitors:
        return f"[ERROR] No competitors stored for {ticker}."

    return "\n".join(competitors)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    print(find_competitors_hardcoded(t))
