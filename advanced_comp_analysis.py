"""
advanced_comp_analysis.py
─────────────────────────
Extends compare_competitors with a full Competitor SWOT & White Space Analysis.
Returns a guaranteed-valid Python dict (JSON-serialisable) — never a raw string.

For the target company and each competitor:
  1. Financial metrics (yfinance)
  2. Recent news signals (yfinance / Yahoo Finance)
  3. Hiring signals (FMP historical employee count + key executives)
  4. Source-backed SWOT per company  — generated with OpenAI JSON mode
  5. White Space & strategic gap analysis — generated with OpenAI JSON mode

JSON guarantee: every LLM call uses response_format={"type":"json_object"},
which forces token-level valid JSON. Each result is also validated with
json.loads() and falls back to a safe error dict if something still goes wrong.

Output schema
─────────────
{
  "target": str,
  "generated_at": str,           # ISO timestamp
  "tickers": [str, ...],         # target first, then competitors
  "metrics": {
    "<TICKER>": {
      "Price": str, "Mkt Cap ($B)": str, "P/E (TTM)": str,
      "Fwd P/E": str, "Rev Growth": str, "Profit Margin": str
    }, ...
  },
  "swot": {
    "<TICKER>": {
      "strengths":    [{"text": str, "source_name": str, "source_url": str}, ...],
      "weaknesses":   [...],
      "opportunities":[...],
      "threats":      [...],
      "hiring_insight": str
    }, ...
  },
  "white_space": {
    "opportunities": [
      {"name": str, "gap": str, "signal": str, "action": str}, ...
    ],
    "watch_out": {"competitor": str, "signal": str, "description": str}
  }
}

Usage
─────
  CLI : python advanced_comp_analysis.py AAPL
        → prints pretty-printed JSON

  API : from advanced_comp_analysis import advanced_comp_analysis
        result = advanced_comp_analysis("AAPL")   # returns dict
        import json; print(json.dumps(result, indent=2))
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

from competitor_agent import get_competitors

load_dotenv()

FMP_KEY       = os.getenv("FMP_API_KEY")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FMP_BASE = "https://financialmodelingprep.com/stable"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_json_call(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 900,
    fallback: dict | None = None,
) -> dict:
    """
    Call OpenAI with JSON mode enabled.
    Returns a parsed dict. On any failure, returns `fallback` (or {"error": ...}).
    JSON mode guarantees the model emits valid JSON at the token level.
    The json.loads() call is a final safety net.
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        return fallback if fallback is not None else {"error": str(exc)}


# ── Financial metrics ──────────────────────────────────────────────────────────

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


def _fetch_metrics(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        result = {label: _fmt(info.get(field), label) for label, field in YF_FIELDS.items()}
    except Exception:
        result = {label: "N/A" for label in YF_FIELDS}

    return result


# ── News signals — yfinance ────────────────────────────────────────────────────

def _fetch_news(ticker: str, limit: int = 8) -> list[dict]:
    """Return recent Yahoo Finance news items as clean dicts."""
    try:
        raw = yf.Ticker(ticker).news or []
        items = []
        for n in raw[:limit]:
            c = n.get("content", {})
            items.append({
                "title":   c.get("title", ""),
                "summary": c.get("summary", "")[:300],
                "url":     c.get("canonicalUrl", {}).get("url", ""),
                "source":  c.get("provider", {}).get("displayName", ""),
                "date":    (c.get("pubDate") or "")[:10],
            })
        return items
    except Exception:
        return []


def _format_news(items: list[dict]) -> str:
    if not items:
        return "No recent news available."
    lines = []
    for i, n in enumerate(items, 1):
        lines.append(
            f"{i}. [{n['source']}] {n['title']} ({n['date']})\n"
            f"   {n['summary']}\n"
            f"   URL: {n['url'] or 'N/A'}"
        )
    return "\n\n".join(lines)


# ── Hiring signals — FMP ───────────────────────────────────────────────────────

def _fetch_employee_trend(ticker: str) -> str:
    try:
        data = requests.get(
            f"{FMP_BASE}/historical-employee-count",
            params={"symbol": ticker, "apikey": FMP_KEY},
            timeout=10,
        ).json()
        if not isinstance(data, list) or not data:
            return "No employee count data."
        data.sort(key=lambda x: x.get("periodOfReport", ""))
        rows = data[-4:]
        lines = [
            f"  {r.get('periodOfReport','N/A')}: "
            f"{r.get('employeeCount', 0):,} employees (filed {r.get('filingDate','N/A')})"
            for r in rows
        ]
        if len(rows) >= 2:
            first = rows[0].get("employeeCount") or 0
            last  = rows[-1].get("employeeCount") or 0
            if first > 0:
                pct  = (last - first) / first * 100
                word = "grew" if pct > 0 else "shrank"
                lines.append(
                    f"  → Headcount {word} {abs(pct):.1f}% "
                    f"from {rows[0]['periodOfReport']} to {rows[-1]['periodOfReport']}"
                )
        return "\n".join(lines)
    except Exception:
        return "Employee count unavailable."


def _fetch_executives(ticker: str) -> str:
    try:
        data = requests.get(
            f"{FMP_BASE}/key-executives",
            params={"symbol": ticker, "apikey": FMP_KEY},
            timeout=10,
        ).json()
        if not isinstance(data, list) or not data:
            return "No executive data."
        return "\n".join(
            f"  {e.get('name','N/A')} — {e.get('title','N/A')} (since {e.get('titleSince') or 'N/A'})"
            for e in data[:6]
        )
    except Exception:
        return "Executive data unavailable."


# ── SWOT — JSON mode ───────────────────────────────────────────────────────────

_SWOT_FALLBACK = {
    "strengths":     [{"text": "Data unavailable", "source_name": "", "source_url": ""}],
    "weaknesses":    [{"text": "Data unavailable", "source_name": "", "source_url": ""}],
    "opportunities": [{"text": "Data unavailable", "source_name": "", "source_url": ""}],
    "threats":       [{"text": "Data unavailable", "source_name": "", "source_url": ""}],
}


def _generate_swot(ticker: str, news_text: str, hiring_text: str) -> dict:
    prompt = f"""You are a sell-side equity analyst. Generate a SWOT for {ticker}.

RECENT NEWS (Yahoo Finance):
{news_text}

HIRING & TALENT SIGNALS (SEC filings + FMP):
{hiring_text}

Rules:
- 1 bullet per quadrant (4 total).
- Every bullet must be grounded in a specific event or data point from the feeds above.
- No generic filler (e.g. "strong brand"). Be precise.
- Use employee headcount direction (growing = strength, shrinking = weakness).
- Use executive tenure to infer strategic priorities.

Return ONLY a JSON object matching this exact schema:
{{
  "strengths":     [{{"text": "...", "source_name": "...", "source_url": "..."}}],
  "weaknesses":    [{{"text": "...", "source_name": "...", "source_url": "..."}}],
  "opportunities": [{{"text": "...", "source_name": "...", "source_url": "..."}}],
  "threats":       [{{"text": "...", "source_name": "...", "source_url": "..."}}]
}}
"""
    return _safe_json_call(prompt, model="gpt-4o", max_tokens=900, fallback=_SWOT_FALLBACK)


# ── White space — JSON mode ────────────────────────────────────────────────────

_WS_FALLBACK = {
    "opportunities": [{"name": "Data unavailable", "gap": "", "signal": "", "action": ""}],
    "watch_out":     {"competitor": "", "signal": "", "description": "Data unavailable"},
}


def _generate_white_space(target: str, swot_blocks: dict[str, dict]) -> dict:
    competitor_swots = "\n\n".join(
        f"=== {ticker} ===\n{json.dumps(block, indent=2)}"
        for ticker, block in swot_blocks.items()
        if ticker != target
    )

    target_swot_json = json.dumps(swot_blocks.get(target, {}), indent=2)

    prompt = f"""You are a strategy consultant advising {target}'s C-suite.

Below are SWOT analyses (in JSON) for {target}'s top competitors, grounded in
real news and headcount data. Identify WHITE SPACE — underserved segments,
capability vacuums, or strategic voids {target} can capture.

COMPETITOR SWOTS:
{competitor_swots}

{target} SWOT (for reference):
{target_swot_json}

Rules:
- Exactly 2 opportunities and 1 watch_out.
- Every field grounded in specific evidence from the SWOTs above.
- No platitudes.

Return ONLY a JSON object matching this exact schema:
{{
  "opportunities": [
    {{
      "name":   "Short descriptive title (e.g. Enterprise AI Integration Vacuum)",
      "gap":    "Which competitor weakness/threat creates this opening?",
      "signal": "Evidence from the data that the window is open (or that {target} is already moving)",
      "action": "Concrete move {target} should make"
    }}
  ],
  "watch_out": {{
    "competitor":  "Ticker of the competitor most likely to close a window",
    "signal":      "The specific hiring or news signal that indicates this",
    "description": "What they are doing and which white-space window it threatens"
  }}
}}
"""
    return _safe_json_call(prompt, model="gpt-4o", max_tokens=1000, fallback=_WS_FALLBACK)


# ── Master function ────────────────────────────────────────────────────────────

def advanced_comp_analysis(target: str) -> dict:
    """
    Run the full analysis and return a guaranteed-valid dict.
    Every LLM call uses JSON mode; every result is validated with json.loads().
    """
    target      = target.upper().strip()
    competitors = get_competitors(target)[:5]
    all_tickers = [target] + competitors

    print(f"[1/4] Fetching financial metrics for {all_tickers}...")
    with ThreadPoolExecutor(max_workers=len(all_tickers)) as ex:
        futures = {ex.submit(_fetch_metrics, t): t for t in all_tickers}
        metrics: dict[str, dict] = {}
        for fut in as_completed(futures):
            metrics[futures[fut]] = fut.result()

    print(f"[2/4] Fetching news + hiring signals...")
    def _fetch_signals(t: str) -> tuple[str, str, str]:
        news    = _format_news(_fetch_news(t))
        hiring  = (
            f"EMPLOYEE HEADCOUNT TREND:\n{_fetch_employee_trend(t)}\n\n"
            f"KEY EXECUTIVES:\n{_fetch_executives(t)}"
        )
        return t, news, hiring

    with ThreadPoolExecutor(max_workers=len(all_tickers)) as ex:
        futures2 = {ex.submit(_fetch_signals, t): t for t in all_tickers}
        news_texts:   dict[str, str] = {}
        hiring_texts: dict[str, str] = {}
        for fut in as_completed(futures2):
            t, news, hiring = fut.result()
            news_texts[t]   = news
            hiring_texts[t] = hiring
            print(f"       {t} signals done")

    print(f"[3/4] Generating source-backed SWOTs (JSON mode)...")
    with ThreadPoolExecutor(max_workers=len(all_tickers)) as ex:
        futures3 = {
            ex.submit(_generate_swot, t, news_texts[t], hiring_texts[t]): t
            for t in all_tickers
        }
        swot: dict[str, dict] = {}
        for fut in as_completed(futures3):
            t = futures3[fut]
            swot[t] = fut.result()
            print(f"       {t} SWOT done")

    print(f"[4/4] Synthesising white space analysis (JSON mode)...")
    white_space = _generate_white_space(target, swot)

    return {
        "target":       target,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers":      all_tickers,
        "metrics":      metrics,
        "swot":         swot,
        "white_space":  white_space,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = advanced_comp_analysis(ticker)
    print(json.dumps(result, indent=2))
