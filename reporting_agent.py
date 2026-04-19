"""Standalone helper for invoking the reporting agent with arbitrary prompts."""

from __future__ import annotations

import argparse
from typing import Optional

from analyst import analyze_filings, analyze_financials
from deepagents import create_deep_agent
from dcf import find_dcf_tool
from langchain_openai import ChatOpenAI
from reporting_pipeline import reporting_tools

TOOLS = reporting_tools
SYSTEM_PROMPT = """
You are a professional financial analyst tasked with completely answering any prompts given to you.

You have access to the full suite of equity reporting tools, including:
- analyze_filings
- find_dcf_tool
- analyze_financials
- valuation_tool
- analyze_news
- get_daily_yf_tool
- analyze_weighted_synthesis
- calculate_moving_average_tool
- calculate_trend_regime_tool
- calculate_rsi_tool
- calculate_atr_tool
- analyze_divergence_tool

Provide accurate, concise, data-driven reasoning. Always use analyze_weighted_synthesis when performing multi-signal evaluations.
"""


def build_agent():
    """Create a DeepAgent instance with the reporting tools."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=30)
    return create_deep_agent(model=llm, system_prompt=SYSTEM_PROMPT, tools=TOOLS)


def format_prompt(company: str, year: str, ticker: Optional[str] = None, prompt: Optional[str] = None) -> str:
    """Return either the custom prompt or a descriptive default template."""

    if prompt and prompt.strip():
        return prompt

    ticker_clause = f" (ticker: {ticker.upper()})" if ticker else ""
    return (
        f"Generate a professional equity outlook for {company}{ticker_clause} covering {year}. "
        "Blend filings, valuation work, and recent price action to explain upside, downside, and key catalysts."
    )


def run_agent(company: str, year: str, ticker: Optional[str], prompt: Optional[str]) -> str:
    agent = build_agent()
    final_prompt = format_prompt(company, year, ticker, prompt)
    response = agent.invoke({"messages": [{"role": "user", "content": final_prompt}]})
    return response["messages"][-1].content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke the reporting agent directly.")
    parser.add_argument("--company", required=True, help="Company name to analyze.")
    parser.add_argument("--year", required=True, help="Fiscal/forecast year to focus on.")
    parser.add_argument("--ticker", help="Optional stock ticker symbol.")
    parser.add_argument(
        "--prompt",
        help="Optional custom natural-language prompt. When omitted, a generic template is used.",
    )
    args = parser.parse_args()

    print(run_agent(args.company, args.year, args.ticker, args.prompt))
