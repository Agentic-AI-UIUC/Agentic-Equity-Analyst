import uuid, os, pytz, datetime, math
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any
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

_COLLECTION = None

def get_financial_collection():
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    
    db = os.getenv("CHROMADB")
    api_key = os.getenv("CHROMADB_API_KEY")
    tenant = os.getenv("CHROMADB_TENANT")
    
    if not all([db, api_key, tenant]):
        return None

    from langchain_chroma import Chroma
    _COLLECTION = Chroma(
        database=db,
        collection_name="financial_data",
        embedding_function=embeddings,
        chroma_cloud_api_key=api_key,
        tenant=tenant,
    )
    return _COLLECTION


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

ny_tz = pytz.timezone("America/New_York")

def get_daily_yf(company: str, symbol: str, days: int = 365):
    """
    Fetch daily close data from Yahoo Finance for the specified number of days.

    Args:
        company: Company name
        symbol: Stock ticker symbol
        days: Number of days of historical data to fetch (default: 365)
    """
    ticker = yf.Ticker(symbol)
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    data = ticker.history(start=start_date, end=end_date,
        interval="1d",
        auto_adjust=True
    )

    texts = []
    metadatas = []
    ids = []

    for date_index, row in data.iterrows():
        # Convert timestamp to date
        trade_date = date_index.date()

        text = (
            f"date={trade_date.isoformat()} "
            f"Open={row['Open']:.2f} High={row['High']:.2f} Low={row['Low']:.2f} Close={row['Close']:.2f} "
            f"Volume={int(row['Volume'])} Dividends={row.get('Dividends', 0.0)} "
            f"StockSplits={row.get('Stock Splits', 0.0)}"
        )
        texts.append(text)

        meta = {
            "symbol": symbol,
            "company": company,
            "date_retrieved": date.today().isoformat(),
            "time_retrieved": datetime.datetime.now(timezone.utc).astimezone(ny_tz).strftime("%H:%M:%S"),
            "date_published": trade_date.isoformat(),
            "source": "yf_daily",
            "url": f"https://finance.yahoo.com/quote/{symbol}",
        }
        metadatas.append(meta)
        ids.append(str(uuid.uuid4()))

    collection = get_financial_collection()
    if not collection:
        logging.error("ChromaDB not configured. Cannot add texts.")
        return

    for t_chunk, m_chunk, id_chunk in zip(chunked(texts, BATCH), chunked(metadatas, BATCH), chunked(ids, BATCH)):
        collection.add_texts(texts=t_chunk, metadatas=m_chunk, ids=id_chunk)

@tool
def get_daily_yf_tool(company: str, symbol: str, days: int = 365):
    """
    Get daily historical stock price data (OHLCV - Open, High, Low, Close, Volume) from Yahoo Finance.
    Fetches daily closing prices for the specified number of days (default: 365 days / 1 year).
    This provides access to historical daily data for technical and fundamental analysis.

    Args:
        company: Company name
        symbol: Stock ticker symbol
        days: Number of days of historical data to fetch (default: 365)
    """
    return get_daily_yf(company, symbol, days)

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


def get_normalized_technical_score(ticker: str) -> Dict[str, Any]:
    """
    Returns a normalized technical score (0.0-1.0) based on multi-horizon Z-scores.
    Uses 50d, 200d, and 365d moving averages blended: (0.5, 0.3, 0.2).
    """
    try:
        import yfinance as yf
        from datetime import date, timedelta
        import math
        stock = yf.Ticker(ticker)
        
        info = stock.info
        current_price = info.get("currentPrice")
        if not current_price:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                return {"score": 0.5, "confidence": 0.0, "error": "No price data"}
        
        # Get history for MA and StdDev calculations
        end_date = date.today()
        start_date = end_date - timedelta(days=650)
        full_hist = stock.history(start=start_date, end=end_date)
        
        if full_hist.empty or len(full_hist) < 365:
             return {
                 "score": 0.5, 
                 "confidence": 0.0, 
                 "error": f"Insufficient history (need 365 days, got {len(full_hist)})"
             }

        prices = full_hist['Close']
        
        def calc_z_sigmoid(window: int):
            if len(prices) < window: return 0.5, 0.0
            subset = prices.tail(window)
            ma = subset.mean()
            std = subset.std()
            if std == 0: return 0.5, 0.0
            z = (current_price - ma) / std
            # Map Z to (0,1) using Sigmoid
            normalized = 1 / (1 + math.exp(-z))
            return normalized, z, ma, std

        # Calculate for each horizon
        s_norm, s_z, s_ma, s_std = calc_z_sigmoid(50)
        m_norm, m_z, m_ma, m_std = calc_z_sigmoid(200)
        l_norm, l_z, l_ma, l_std = calc_z_sigmoid(365)
        
        # Weighted Composite Score
        composite_score = (0.5 * s_norm) + (0.3 * m_norm) + (0.2 * l_norm)
        
        # Volatility & Confidence
        # Volatility as coefficient of variation for the long horizon
        volatility = l_std / l_ma if l_ma != 0 else 0.5
        trend_strength = abs(l_z)
        
        # confidence = min(1.0, trend_strength / 3) * (1 - volatility)
        # We cap confidence but allow it to scale with deviation
        confidence = min(1.0, trend_strength / 3.0) * (1.0 - min(volatility * 2, 0.7))
        
        return {
            "score": round(composite_score, 3),
            "confidence": round(confidence, 3),
            "horizons": {
                "50d": {"score": round(s_norm, 3), "z": round(s_z, 3)},
                "200d": {"score": round(m_norm, 3), "z": round(m_z, 3)},
                "365d": {"score": round(l_norm, 3), "z": round(l_z, 3)}
            },
            "volatility": round(volatility, 3),
            "current_price": round(float(current_price), 2)
        }
    except Exception as e:
        return {"score": 0.5, "confidence": 0.0, "error": str(e)}

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

@tool
def calculate_rsi_tool(ticker: str, period: int = 14) -> str:
    """
    Calculate the Relative Strength Index (RSI) for a given stock ticker.

    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    - RSI > 65: Overbought (price may pullback)
    - RSI < 45: Oversold (price may bounce)
    - Otherwise: Neutral

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        Formatted string with RSI value and interpretation
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        # Fetch enough data to calculate RSI with buffer for weekends/holidays
        start_date = end_date - timedelta(days=period + 50)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"Error: No data found for ticker {ticker}"

        if len(hist) < period + 1:
            return f"Error: Insufficient data for RSI calculation. Need at least {period + 1} days, only {len(hist)} days available."

        # Calculate price changes
        delta = hist['Close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gains / avg_losses

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Get the most recent RSI value
        current_rsi = rsi.iloc[-1]
        current_price = hist['Close'].iloc[-1]

        # Determine market condition and advice
        if current_rsi > 65:
            condition = "OVERBOUGHT"
            advice = "The stock is currently overbought, which suggests that the price may pullback or consolidate in the near term. Consider waiting for a better entry point or taking profits if you hold a position."
        elif current_rsi < 45:
            condition = "OVERSOLD"
            advice = "The stock is currently oversold, which suggests that the price may bounce or recover in the near term. This could present a buying opportunity for investors looking for an entry point."
        else:
            condition = "NEUTRAL"
            advice = "The stock is in a neutral zone, indicating balanced buying and selling pressure. Monitor for trend signals before making trading decisions."

        return (
            f"RSI Analysis for {ticker}:\n"
            f"Current Price: ${current_price:.2f}\n"
            f"RSI ({period}-period): {current_rsi:.2f}\n"
            f"Condition: {condition}\n"
            f"Interpretation: {advice}"
        )

    except Exception as e:
        return f"Error calculating RSI for {ticker}: {str(e)}"

@tool
def calculate_atr_tool(ticker: str, period: int = 14) -> str:
    """
    Calculate the Average True Range (ATR) for a given stock ticker.

    ATR measures market volatility by calculating the average of true ranges over a period.
    - Low ATR: Quiet market with less volatility
    - High ATR: Volatile market with larger price swings

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
        period: Number of periods for ATR calculation (default: 14)

    Returns:
        Formatted string with ATR value, interpretation, and volatility assessment
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        # Fetch enough data to calculate ATR with buffer for weekends/holidays
        start_date = end_date - timedelta(days=period + 50)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"Error: No data found for ticker {ticker}"

        if len(hist) < period + 1:
            return f"Error: Insufficient data for ATR calculation. Need at least {period + 1} days, only {len(hist)} days available."

        # Calculate True Range (TR)
        # TR = max[(High - Low), abs(High - Previous Close), abs(Low - Previous Close)]
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()

        true_range = high_low.combine(high_close, max).combine(low_close, max)

        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=period, adjust=False).mean()

        # Get the most recent ATR value
        current_atr = atr.iloc[-1]
        current_price = hist['Close'].iloc[-1]

        # Calculate ATR as percentage of current price for better interpretation
        atr_percentage = (current_atr / current_price) * 100

        # Determine volatility level
        # Generally, ATR % < 2% is low volatility, 2-5% is moderate, > 5% is high
        if atr_percentage < 2.0:
            volatility_level = "LOW"
            interpretation = "The market is relatively quiet with low volatility. Price movements are contained, which may indicate consolidation or low trading activity. This environment may suit range-bound trading strategies."
        elif atr_percentage < 5.0:
            volatility_level = "MODERATE"
            interpretation = "The market shows moderate volatility with normal price fluctuations. This balanced environment is typical for most trading conditions and allows for various trading strategies."
        else:
            volatility_level = "HIGH"
            interpretation = "The market is volatile with significant price swings. High volatility can present both opportunities and risks. Consider wider stop-losses and be prepared for larger price movements. This environment may suit momentum or breakout strategies."

        return (
            f"ATR Analysis for {ticker}:\n"
            f"Current Price: ${current_price:.2f}\n"
            f"ATR ({period}-period): ${current_atr:.2f}\n"
            f"ATR as % of Price: {atr_percentage:.2f}%\n"
            f"Volatility Level: {volatility_level}\n"
            f"Interpretation: {interpretation}"
        )

    except Exception as e:
        return f"Error calculating ATR for {ticker}: {str(e)}"

