"""
competitor_agent.py
───────────────────
Returns the top 5 widely-accepted competitors for any S&P 500 company.
Also provides utilities to fetch S&P 500 and ADR ticker universes.

Source: competitors.json — hand-curated peer groups reflecting
what is broadly accepted on financial sites, analyst reports, and
general public knowledge. Fully deterministic — no API calls, no ML.

Usage
─────
  CLI (competitors) :  python competitor_agent.py GOOGL
  CLI (ticker list) :  python competitor_agent.py --list-tickers
  CLI (ADR list)    :  python competitor_agent.py --list-tickers --min-cap 50
  Tool              :  from competitor_agent import competitor_tool_hardcoded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

WIKI_SP500_URL   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
ADR_API_BASE     = "https://api.markitdigital.com/jpmadr-public/v1/drUniverse"
TARGET_EXCHANGES = {"NYSE", "NASDAQ"}


# ─── TICKER UNIVERSE FUNCTIONS ────────────────────────────────────────────────

def fetch_sp500_tickers() -> dict[str, str]:
    """Return {ticker: company_name} for all current S&P 500 constituents."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    resp = requests.get(WIKI_SP500_URL, timeout=20,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    result: dict[str, str] = {}
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 2:
            ticker = cols[0].get_text(strip=True).replace(".", "-")
            name   = cols[1].get_text(strip=True)
            if ticker:
                result[ticker] = name
    return result


def fetch_adr_tickers(min_market_cap_b: float = 30.0) -> list[dict]:
    """
    Page through the DR Universe API, keep only NYSE / NASDAQ tickers
    above the market-cap floor, sorted by market cap descending.
    """
    min_cap   = min_market_cap_b * 1_000_000_000
    page_size = 500
    offset    = 0
    collected: list[dict] = []

    print(f"Fetching ADR universe from adr.com (NYSE/NASDAQ, >= ${min_market_cap_b:.0f}B)...")
    while True:
        resp = requests.get(
            ADR_API_BASE,
            params={"limit": page_size, "offset": offset},
            timeout=20,
        ).json()

        data = resp.get("data", {})
        rows = data.get("items", [])
        if not rows:
            break

        for row in rows:
            exchange   = (row.get("exchange") or "").upper().strip()
            market_cap = row.get("globalMarketCap") or 0
            ticker     = (row.get("ticker") or "").upper().strip()
            if exchange in TARGET_EXCHANGES and market_cap >= min_cap and ticker:
                collected.append({
                    "ticker":   ticker,
                    "name":     row.get("name", ticker),
                    "exchange": exchange,
                    "country":  row.get("country", ""),
                    "mktcap":   market_cap,
                })

        if not data.get("pagination", {}).get("hasMore"):
            break
        offset += page_size

    collected.sort(key=lambda x: x["mktcap"], reverse=True)
    return collected


def print_ticker_universe(min_cap_b: float = 30.0) -> None:
    sp500 = fetch_sp500_tickers()
    adrs  = fetch_adr_tickers(min_cap_b)

    adr_map  = {r["ticker"]: r for r in adrs}
    combined = sorted(set(sp500) | set(adr_map))

    print(f"\n{'='*60}")
    print(f"S&P 500  ({len(sp500)} tickers)")
    print(f"{'='*60}")
    for ticker, name in sorted(sp500.items()):
        print(f"  {ticker:<10} {name}")

    print(f"\n{'='*60}")
    print(f"ADR  ({len(adrs)} tickers, sorted by market cap)")
    print(f"{'='*60}")
    print(f"  {'Ticker':<10} {'Market Cap':>12}  {'Exchange':<8} {'Country':<20} Name")
    print(f"  {'-'*9} {'-'*12}  {'-'*7} {'-'*19} {'-'*30}")
    for r in adrs:
        cap_str = f"${r['mktcap']/1e9:.1f}B"
        print(f"  {r['ticker']:<10} {cap_str:>12}  {r['exchange']:<8} {r['country']:<20} {r['name']}")

    adr_only = [t for t in adr_map if t not in sp500]
    overlap  = [t for t in adr_map if t in sp500]

    print(f"\n{'='*60}")
    print(f"Combined Universe  ({len(combined)} tickers)")
    print(f"{'='*60}")
    print(f"  S&P 500 only : {len(sp500) - len(overlap)}")
    print(f"  ADR only     : {len(adr_only)}")
    print(f"  In both      : {len(overlap)}")
    if overlap:
        print(f"  Overlap tickers: {', '.join(sorted(overlap))}")
    print(f"\n  All tickers: {', '.join(combined)}")

COMPETITORS_FILE = Path(__file__).parent / "competitors.json"

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

    Uses a static JSON database (competitors.json) that encodes broadly
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
    parser = argparse.ArgumentParser(description="Competitor lookup and ticker universe tools")
    parser.add_argument("ticker", nargs="?", default="NVDA", help="Stock ticker to look up competitors for")
    parser.add_argument("--list-tickers", action="store_true", help="Print S&P 500, ADR, and combined ticker universe")
    parser.add_argument("--min-cap", type=float, default=30.0, metavar="BILLION", help="Minimum ADR market cap in billions (default: 30.0)")
    args = parser.parse_args()

    if args.list_tickers:
        print_ticker_universe(min_cap_b=args.min_cap)
    else:
        print(get_competitors(args.ticker))
