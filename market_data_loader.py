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
