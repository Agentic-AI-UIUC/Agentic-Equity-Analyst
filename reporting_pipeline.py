"""High-level orchestration utilities for generating equity research reports."""

from __future__ import annotations

from typing import Optional

from analyst import analyze_filings, analyze_financials, analyze_news, analyze_parser
from deepagents import create_deep_agent
from dcf import find_dcf_tool
from langchain import agents
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from market_data_loader import calculate_moving_average_tool, calculate_trend_regime_tool, calculate_rsi_tool, calculate_atr_tool, get_daily_yf_tool
from pdf_builder import report
from valuation_agent import valuation_tool

LLM_MANAGER = init_chat_model("gpt-5.1", model_provider="openai")
LLM_REPORTER = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=30)

MANAGER_PROMPT = (
    "You are a helpful professional equity analyst. "
    "Run the reporting tool once per request and return the tool's response."
)

REPORTING_PROMPT = """
You are a helpful professional financial analyst tasked with consulting a user.

You have access to these tools:
- analyze_filings: find specific financial metrics of a company in its 10-Q and 10-K filings.
- find_dcf_tool: run a Discounted Cash Flow analysis for a company and year.
- analyze_financials: retrieve financial ticker data for a company.
- valuation_tool: summarize equity research valuation commentary for a company and year.
- analyze_news: extract recent qualitative signals from news coverage.
- get_daily_yf_tool: fetch historical daily stock price data (OHLCV) from Yahoo Finance for up to 365 days of trading history.
- calculate_moving_average_tool: calculate the 365-day moving average for a stock ticker.
- calculate_trend_regime_tool: analyze the trend regime using 50-day and 200-day moving averages to determine if the stock is in a bullish, bearish, or neutral trend.
- calculate_rsi_tool: calculate the Relative Strength Index (RSI) to identify overbought, oversold, or neutral conditions with actionable trading advice.
- calculate_atr_tool: calculate the Average True Range (ATR) to measure market volatility and identify quiet or volatile market conditions.

CRITICAL - TECHNICAL ANALYSIS INTEGRATION:
When performing technical analysis, you MUST use multiple indicators together and synthesize them into unified insights:

1. ALWAYS run multiple technical indicators (RSI, ATR, Moving Averages, Trend Regime) when analyzing a stock's technical position.

2. INTEGRATE the indicators to assess:
   - TREND DIRECTION (Bullish vs Bearish):
     * Compare trend regime (50/200 MA) with RSI momentum
     * Check if moving averages support or contradict RSI signals
     * Determine if indicators confirm each other or show divergence

   - MARKET DYNAMICS (Slow vs Fast Movement):
     * Use ATR to assess volatility levels
     * Correlate volatility with trend strength
     * Identify if high/low volatility supports or conflicts with trend signals

3. PROVIDE UNIFIED CONCLUSIONS:
   - Synthesize all indicators into clear statements like:
     * "Strong bullish case with high conviction" (when trend, MA, and RSI all align bullish)
     * "Bearish with caution" (when indicators show bearish trend but RSI is oversold)
     * "Slow, range-bound movement expected" (when trend is neutral and ATR is low)
     * "Fast, volatile bullish momentum" (when bullish trend combines with high ATR)

   - Highlight CONFIRMATION: "Multiple indicators confirm [direction]"
   - Flag DIVERGENCE: "RSI shows [X] while trend indicates [Y], suggesting [interpretation]"

4. COMPARE AND CONTRAST:
   - Explicitly state agreement or disagreement between indicators
   - Assess strength of conviction based on how many indicators align
   - Provide context: "The bullish trend is supported by [X, Y] but tempered by [Z]"

5. ACTIONABLE SYNTHESIS:
   - Give clear, integrated recommendations based on the full technical picture
   - Address both direction (buy/sell/hold) and timing (enter now vs wait)
   - Account for volatility in risk management advice

6. ALWAYS INCLUDE TECHNICAL ANALYSIS IN YOUR FINAL REPORT:
   - Your final report MUST contain a dedicated "Technical Analysis" section
   - This section should present all technical indicator results (Moving Averages, RSI, ATR, Trend Regime)
   - Provide the integrated synthesis and unified conclusions from these indicators
   - Do not omit technical analysis even if fundamental analysis is strong
   - Technical analysis should be a core component of every equity research report you generate

Return accurate, concise, data-driven guidance that integrates all technical indicators into cohesive insights.
"""

reporting_tools = [
    analyze_filings,
    analyze_parser,
    analyze_financials,
    analyze_news,
    valuation_tool,
    find_dcf_tool,
    get_daily_yf_tool,
    calculate_moving_average_tool,
    calculate_trend_regime_tool,
    calculate_rsi_tool,
    calculate_atr_tool,
]

reporting_agent = agents.create_agent(
    model=LLM_REPORTER,
    system_prompt=REPORTING_PROMPT,
    tools=reporting_tools,
)


def _normalize_message_payload(message) -> str:
    """Best-effort conversion from LangChain message payloads into text."""

    content = getattr(message, "text", None) or getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    return str(content)


@tool
def create_report(request: str) -> str:
    """Invoke the reporting agent with a natural-language request."""

    res = reporting_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return _normalize_message_payload(res["messages"][-1])


def _summarize_prompt(prompt: str) -> str:
    """Use the manager LLM to lightly compress the user prompt for efficiency."""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful summarizer. Focus on clarity and keep the prompt under 60 words."
            )
        ),
        HumanMessage(
            content=f"Summarize and reformulate the following request without losing intent: {prompt}"
        ),
    ]

    try:
        summary_message = LLM_MANAGER.invoke(messages)
        return _normalize_message_payload(summary_message)
    except Exception:
        return prompt


manager_agent = create_deep_agent(
    model=LLM_MANAGER,
    tools=[create_report],
    system_prompt=MANAGER_PROMPT,
)


def _invoke_manager(prompt: str) -> str:
    """Send the condensed instruction to the manager agent and return the report text."""

    response = manager_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return _normalize_message_payload(response["messages"][-1])


DEFAULT_PROMPT_TEMPLATE = (
    "Create a professional equity research style outlook for {company}{ticker_clause} covering {year}. "
    "Highlight financial performance, valuation, major risks, catalysts, and data-supported insights. "
    "\n\nIMPORTANT: Include a comprehensive TECHNICAL ANALYSIS section that integrates:\n"
    "- Moving Averages (365-day, 50-day, 200-day)\n"
    "- RSI (Relative Strength Index) with overbought/oversold analysis\n"
    "- ATR (Average True Range) for volatility assessment\n"
    "- Trend Regime analysis (bullish/bearish/neutral)\n"
    "Synthesize these indicators to provide a unified technical outlook on direction (bullish/bearish) "
    "and market dynamics (slow/fast movement, volatility). Highlight confirmations and divergences between indicators."
)


def build_prompt(
    company: str,
    year: str,
    ticker: Optional[str] = None,
    custom_prompt: Optional[str] = None,
) -> str:
    """Return either the user-provided prompt or a descriptive default template."""

    if custom_prompt and custom_prompt.strip():
        return custom_prompt

    ticker_clause = f" (ticker: {ticker.upper()})" if ticker else ""
    return DEFAULT_PROMPT_TEMPLATE.format(
        company=company,
        ticker_clause=ticker_clause,
        year=year,
    )


def generate_financial_report(
    *,
    company: str,
    ticker: Optional[str] = None,
    year: str,
    custom_prompt: Optional[str] = None,
    launch_ui: bool = False,
    file_path: str = "report.txt",
) -> str:
    """Run the end-to-end reporting pipeline and persist results to disk."""

    user_prompt = build_prompt(company, year, ticker, custom_prompt)
    condensed_prompt = _summarize_prompt(user_prompt)
    report_text = _invoke_manager(condensed_prompt)
    report(report_text, launch_ui=launch_ui, file_path=file_path)
    return report_text


__all__ = ["generate_financial_report", "build_prompt"]
