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
from market_data_loader import calculate_moving_average_tool
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
- calculate_moving_average_tool: calculate the 365-day moving average for a stock ticker.

Return accurate, concise, data-driven guidance.
"""

reporting_tools = [
    analyze_filings,
    analyze_parser,
    analyze_financials,
    analyze_news,
    valuation_tool,
    find_dcf_tool,
    calculate_moving_average_tool,
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
    "Highlight financial performance, valuation, major risks, catalysts, and data-supported insights."
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
