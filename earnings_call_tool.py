"""Live-fetch earnings call transcript analysis using FMP API and GPT-4o."""

import os

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

fmp_key = os.getenv("FMP_API_KEY")
model = init_chat_model("gpt-4o", model_provider="openai")

FMP_TRANSCRIPT_URL = "https://financialmodelingprep.com/stable/earning-call-transcript"


def fetch_transcript(ticker: str, year: int, quarter: int) -> dict | None:
    """Fetch a single earnings call transcript from FMP API."""
    params = {"symbol": ticker, "year": year, "quarter": quarter, "apikey": fmp_key}
    try:
        resp = requests.get(FMP_TRANSCRIPT_URL, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
    except requests.RequestException:
        pass
    return None


def analyze_transcript(transcript_content: str, query: str) -> str:
    """Use GPT-4o to parse and analyze a raw transcript against a query."""
    system_prompt = """You are a financial analyst parsing an earnings call transcript.

Extract and organize:
1. **Prepared Remarks**: Key statements from executives (identify CEO, CFO, etc.)
2. **Q&A Highlights**: Analyst questions and management responses
3. **Forward Guidance**: Revenue, margin, growth projections
4. **Sentiment Signals**: Management confidence, concerns, hedging language

Distinguish executive statements from analyst questions. Include speaker names."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query}\n\nTranscript:\n{transcript_content}"),
    ]
    return model.invoke(messages).content


@tool
def analyze_earnings_calls(ticker: str, year: str) -> str:
    """
    Analyze earnings call transcripts for executive commentary and analyst Q&A.
    Extracts forward-looking guidance, management sentiment, and analyst concerns.
    Distinguishes CEO/CFO prepared remarks from analyst questions.
    Takes two arguments: the ticker symbol (e.g. 'NVDA') and the year (formatted as 'XXXX').
    """
    try:
        year_int = int(year)

        for quarter in [4, 3, 2, 1]:
            transcript = fetch_transcript(ticker, year_int, quarter)
            if transcript:
                content = transcript.get("content", "")
                if not content:
                    continue
                header = f"## Earnings Call Analysis: {ticker} Q{quarter} {year}\n\n"
                return header + analyze_transcript(content, f"{ticker} {year} earnings")

        # Try previous year as fallback
        for quarter in [4, 3, 2, 1]:
            transcript = fetch_transcript(ticker, year_int - 1, quarter)
            if transcript:
                content = transcript.get("content", "")
                if not content:
                    continue
                header = f"## Earnings Call Analysis: {ticker} Q{quarter} {year_int - 1}\n\n"
                return header + analyze_transcript(content, f"{ticker} {year_int - 1} earnings")

        return f"No earnings call transcript found for {ticker} in {year} or {year_int - 1}."
    except Exception as e:
        return f"Error analyzing earnings calls: {e}"
