import uuid, os, pytz, datetime
from dotenv import load_dotenv
import yfinance as yf
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from datetime import date, timedelta, timezone
from openai import OpenAI
from langchain.tools import tool 

BATCH = 300

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

collection = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="financial_data", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

ny_tz = pytz.timezone("America/New_York")

def get_daily_yf(company: str, symbol: str):
    ticker = yf.Ticker(symbol)
    data = ticker.history(start= date.today() - timedelta(days=1), end=date.today(),
        interval="1m",
        auto_adjust=True,
        prepost=True
    )

    texts = []
    metadatas = []
    ids = []

    for ts_utc, row in data.iterrows():
        text = (
            f"timestamp_utc={ts_utc.isoformat()} "
            f"Open={row['Open']} High={row['High']} Low={row['Low']} Close={row['Close']} "
            f"Volume={row['Volume']} Dividends={row.get('Dividends', 0.0)} "
            f"StockSplits={row.get('Stock Splits', 0.0)}"
        )
        texts.append(text)

        ts_ny = datetime.datetime.now(ny_tz)
        meta = {
            "symbol": symbol,
            "company": company,
            "date_retrieved": date.today().isoformat(),
            "time_retrieved": datetime.datetime.now(timezone.utc).astimezone(ny_tz).strftime("%H:%M:%S"),
            "date_published": ts_ny.date().isoformat(), 
            "time_published": ts_ny.strftime("%H:%M:%S"),    
            "source": "yf",
            "url": "url",
        }
        metadatas.append(meta)
        ids.append(str(uuid.uuid4()))

    for t_chunk, m_chunk, id_chunk in zip(chunked(texts, BATCH), chunked(metadatas, BATCH), chunked(ids, BATCH)):
        collection.add_texts(texts=t_chunk, metadatas=m_chunk, ids=id_chunk)

@tool
def get_daily_yf_tool(company: str, symbol: str):
    """
    Get financial ticker data for a given company within the last day
    Takes company name and its ticker as the arguments
    """
    return get_daily_yf(company, symbol)

@tool
def calculate_moving_average_tool(ticker: str, days: int = 365) -> str:
    """
    Calculate the moving average for a given stock ticker over a specified number of days.
    Defaults to 365 days if not specified.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
        days: Number of days for moving average calculation (default: 365)

    Returns:
        Formatted string with the moving average value
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        # Add buffer days to account for weekends/holidays
        start_date = end_date - timedelta(days=days + 100)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"Error: No data found for ticker {ticker}"

        if len(hist) < days:
            actual_days = len(hist)
            moving_avg = hist['Close'].mean()
            return f"Warning: Only {actual_days} days of data available. Moving average for period of last {actual_days} days: ${moving_avg:.2f}"

        # Calculate moving average using the most recent 'days' closing prices
        moving_avg = hist['Close'].tail(days).mean()

        return f"Moving average for period of last {days} days: ${moving_avg:.2f}"

    except Exception as e:
        return f"Error calculating moving average for {ticker}: {str(e)}"

@tool
def calculate_trend_regime_tool(ticker: str) -> str:
    """
    Calculate the trend regime for a given stock ticker based on 50-day and 200-day moving averages.

    Determines if the stock is in a bullish or bearish trend based on:
    - Bullish: 50-day MA > 200-day MA AND current price > 200-day MA
    - Bearish: 50-day MA < 200-day MA AND current price < 200-day MA
    - Neutral: Any other combination

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)

    Returns:
        Formatted string with trend regime analysis including current price, 50-day MA, 200-day MA, and trend determination
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        # Fetch enough data to calculate 200-day MA (add buffer for weekends/holidays)
        start_date = end_date - timedelta(days=300)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"Error: No data found for ticker {ticker}"

        if len(hist) < 200:
            return f"Error: Insufficient data for trend regime analysis. Need at least 200 days, only {len(hist)} days available."

        # Get current price (most recent close)
        current_price = hist['Close'].iloc[-1]

        # Calculate 50-day moving average
        ma_50 = hist['Close'].tail(50).mean()

        # Calculate 200-day moving average
        ma_200 = hist['Close'].tail(200).mean()

        # Determine trend regime
        if ma_50 > ma_200 and current_price > ma_200:
            trend = "BULLISH"
            explanation = "The 50-day moving average is above the 200-day moving average, and the current price is above the 200-day moving average, indicating a bullish trend."
        elif ma_50 < ma_200 and current_price < ma_200:
            trend = "BEARISH"
            explanation = "The 50-day moving average is below the 200-day moving average, and the current price is below the 200-day moving average, indicating a bearish trend."
        else:
            trend = "NEUTRAL"
            explanation = "The moving averages show mixed signals, indicating a neutral or transitional trend."

        return (
            f"Trend Regime Analysis for {ticker}:\n"
            f"Current Price: ${current_price:.2f}\n"
            f"50-day Moving Average: ${ma_50:.2f}\n"
            f"200-day Moving Average: ${ma_200:.2f}\n"
            f"Trend: {trend}\n"
            f"{explanation}"
        )

    except Exception as e:
        return f"Error calculating trend regime for {ticker}: {str(e)}"
