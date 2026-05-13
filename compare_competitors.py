"""
compare_competitors.py
──────────────────────
Fetches key financial metrics for a target ticker and its 5 competitors,
then uses gpt-4o-mini to produce a concise investor-facing comparison.

Usage
─────
  CLI : python compare_competitors.py AAPL
  API : from compare_competitors import compare_competitors
        print(compare_competitors("AAPL"))
"""

from __future__ import annotations

import os
import sys

import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

from competitor_agent import get_competitors

load_dotenv()

FMP_KEY = os.getenv("FMP_API_KEY")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Metrics to pull ────────────────────────────────────────────────────────────

YF_FIELDS = {
    "Price":         "currentPrice",
    "Mkt Cap ($B)":  "marketCap",
    "P/E (TTM)":     "trailingPE",
    "Fwd P/E":       "forwardPE",
    "Rev Growth":    "revenueGrowth",
    "Profit Margin": "profitMargins",
}


def _fmt(val, label: str) -> str:
    if val is None:
        return "N/A"
    if label == "Mkt Cap ($B)":
        return f"${val / 1e9:.1f}B"
    if label in ("Rev Growth", "Profit Margin"):
        return f"{val * 100:.1f}%"
    if label == "Price":
        return f"${val:.2f}"
    return f"{val:.2f}"


def _fetch_yf(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {label: info.get(field) for label, field in YF_FIELDS.items()}
    except Exception:
        return {label: None for label in YF_FIELDS}


def _fetch_fmp_beta(ticker: str) -> str:
    """Grab beta from FMP company profile as a supplement."""
    try:
        url = (
            f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
            f"?apikey={FMP_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        beta = data[0].get("beta") if data else None
        return f"{beta:.2f}" if beta else "N/A"
    except Exception:
        return "N/A"


# ── Table builder ──────────────────────────────────────────────────────────────

def _build_table(tickers: list[str], rows: dict[str, dict]) -> str:
    col_w = 13
    header = f"{'Metric':<16}" + "".join(f"{t:>{col_w}}" for t in tickers)
    sep = "-" * len(header)
    lines = [header, sep]
    for metric in list(YF_FIELDS.keys()) + ["Beta"]:
        row = f"{metric:<16}"
        for t in tickers:
            val = rows[t].get(metric, "N/A")
            row += f"{val:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


# ── Main function ──────────────────────────────────────────────────────────────

def compare_competitors(target: str) -> str:
    target = target.upper().strip()
    competitors = get_competitors(target)[:5]
    all_tickers = [target] + competitors

    # Fetch data
    rows: dict[str, dict] = {}
    for t in all_tickers:
        data = _fetch_yf(t)
        rows[t] = {label: _fmt(val, label) for label, val in data.items()}
        rows[t]["Beta"] = _fetch_fmp_beta(t)  # already a formatted string

    table = _build_table(all_tickers, rows)

    # LLM summary
    prompt = (
        f"You are a concise equity analyst. Below is a financial comparison table "
        f"for {target} and its top 5 competitors.\n\n"
        f"{table}\n\n"
        "In exactly 2 bullet points, highlight the 2 most critical takeaways an investor "
        "should know before deciding whether to invest in "
        f"{target} vs its peers. Be direct. No fluff."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    summary = response.choices[0].message.content.strip()

    output = (
        f"COMPETITOR COMPARISON — {target}\n"
        f"{'=' * 60}\n\n"
        f"{table}\n\n"
        f"KEY TAKEAWAYS\n"
        f"{'-' * 40}\n"
        f"{summary}\n"
    )
    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(compare_competitors(ticker))
