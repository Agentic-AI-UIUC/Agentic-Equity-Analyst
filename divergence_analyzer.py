import yfinance as yf
from datetime import date, timedelta
from langchain.tools import tool
from analyst_ratings_loader import load_analyst_ratings


def calculate_rsi(ticker: str, period: int = 14, lookback_days: int = 100):
    """
    Calculate RSI and return bullish/bearish/neutral signal.

    Returns:
        dict with 'value', 'condition', and 'score'
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < period + 1:
            return {"value": None, "condition": "INSUFFICIENT_DATA", "score": 0}

        # Calculate price changes
        delta = hist['Close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Determine score
        # RSI < 45 = OVERSOLD = bullish signal = +1
        # RSI > 65 = OVERBOUGHT = bearish signal = -1
        # Otherwise = NEUTRAL = 0
        if current_rsi < 45:
            condition = "OVERSOLD"
            score = 1  # Bullish
        elif current_rsi > 65:
            condition = "OVERBOUGHT"
            score = -1  # Bearish
        else:
            condition = "NEUTRAL"
            score = 0

        return {
            "value": round(current_rsi, 2),
            "condition": condition,
            "score": score
        }

    except Exception as e:
        return {"value": None, "condition": f"ERROR: {str(e)}", "score": 0}


def calculate_trend_regime(ticker: str, lookback_days: int = 300):
    """
    Calculate trend regime based on 50-day and 200-day moving averages.

    Returns:
        dict with 'ma_50', 'ma_200', 'current_price', 'trend', and 'score'
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < 200:
            return {
                "ma_50": None,
                "ma_200": None,
                "current_price": None,
                "trend": "INSUFFICIENT_DATA",
                "score": 0
            }

        current_price = hist['Close'].iloc[-1]
        ma_50 = hist['Close'].tail(50).mean()
        ma_200 = hist['Close'].tail(200).mean()

        # Determine trend
        # BULLISH: 50-day MA > 200-day MA AND price > 200-day MA = +1
        # BEARISH: 50-day MA < 200-day MA AND price < 200-day MA = -1
        # NEUTRAL: Mixed signals = 0
        if ma_50 > ma_200 and current_price > ma_200:
            trend = "BULLISH"
            score = 1
        elif ma_50 < ma_200 and current_price < ma_200:
            trend = "BEARISH"
            score = -1
        else:
            trend = "NEUTRAL"
            score = 0

        return {
            "ma_50": round(ma_50, 2),
            "ma_200": round(ma_200, 2),
            "current_price": round(current_price, 2),
            "trend": trend,
            "score": score
        }

    except Exception as e:
        return {
            "ma_50": None,
            "ma_200": None,
            "current_price": None,
            "trend": f"ERROR: {str(e)}",
            "score": 0
        }


def calculate_technical_score(ticker: str):
    """
    Calculate combined technical score from RSI and Moving Averages.

    Score ranges from -2 to +2:
    - RSI score: -1, 0, or +1
    - MA score: -1, 0, or +1

    Returns:
        dict with RSI data, MA data, total technical score, and normalized score [-1, 1]
    """
    rsi_data = calculate_rsi(ticker)
    ma_data = calculate_trend_regime(ticker)

    total_score = rsi_data["score"] + ma_data["score"]

    # Z-normalize to [-1, 1] range
    normalized_score = total_score / 2.0

    return {
        "rsi": rsi_data,
        "moving_averages": ma_data,
        "total_score": total_score,
        "normalized_score": normalized_score
    }


def calculate_fundamental_score(ticker: str, lookback_days: int = 30):
    """
    Calculate fundamental score based on analyst ratings.

    Score:
    - Strong Buy / Buy = +1 (positive sentiment)
    - Hold = 0 (neutral sentiment)
    - Sell / Strong Sell = -1 (negative sentiment)

    Also considers rating trend (upgrades/downgrades) in the lookback period.

    Returns:
        dict with rating data and fundamental score
    """
    try:
        ratings_data = load_analyst_ratings(ticker)

        if "error" in ratings_data:
            return {
                "rating_label": "ERROR",
                "consensus_rating": None,
                "rating_trend": None,
                "score": 0,
                "error": ratings_data["error"]
            }

        rating_label = ratings_data.get("rating_label", "Unknown")
        consensus = ratings_data.get("consensus_rating")
        trend = ratings_data.get("rating_trend", {})

        # Convert rating to score
        # Strong Buy (1.0-1.5) or Buy (1.5-2.5) = +1
        # Hold (2.5-3.5) = 0
        # Sell (3.5-4.5) or Strong Sell (4.5-5.0) = -1
        if rating_label in ["Strong Buy", "Buy"]:
            score = 1
        elif rating_label == "Hold":
            score = 0
        elif rating_label in ["Sell", "Strong Sell"]:
            score = -1
        else:
            score = 0  # Unknown

        return {
            "rating_label": rating_label,
            "consensus_rating": consensus,
            "rating_trend": trend,
            "score": score
        }

    except Exception as e:
        return {
            "rating_label": "ERROR",
            "consensus_rating": None,
            "rating_trend": None,
            "score": 0,
            "error": str(e)
        }


def detect_divergence(technical_score: float, fundamental_score: float):
    """
    Detect divergence between technical and fundamental signals.

    Args:
        technical_score: Normalized technical score in range [-1, 1]
        fundamental_score: Normalized fundamental score in range [-1, 1]

    Returns:
        dict with divergence type and interpretation
    """
    if technical_score > 0 and fundamental_score < 0:
        return {
            "type": "BEARISH_DIVERGENCE",
            "description": "Technical indicators are bullish but fundamentals (analyst sentiment) are bearish. Price may be overextended.",
            "signal": "CAUTION - Technical strength not supported by fundamentals"
        }
    elif technical_score < 0 and fundamental_score > 0:
        return {
            "type": "BULLISH_DIVERGENCE",
            "description": "Technical indicators are bearish but fundamentals (analyst sentiment) are bullish. Price may be oversold relative to value.",
            "signal": "OPPORTUNITY - Fundamental strength not reflected in technicals"
        }
    elif technical_score > 0 and fundamental_score > 0:
        return {
            "type": "BULLISH_CONVERGENCE",
            "description": "Both technical and fundamental indicators are bullish. Strong buy signal.",
            "signal": "STRONG BUY - Technical and fundamental alignment"
        }
    elif technical_score < 0 and fundamental_score < 0:
        return {
            "type": "BEARISH_CONVERGENCE",
            "description": "Both technical and fundamental indicators are bearish. Strong sell signal.",
            "signal": "STRONG SELL - Technical and fundamental alignment"
        }
    else:
        return {
            "type": "NEUTRAL_MIXED",
            "description": "Mixed or neutral signals from technical and fundamental indicators.",
            "signal": "HOLD - Wait for clearer signals"
        }


def analyze_divergence_for_period(ticker: str, period_name: str, lookback_days: int):
    """
    Analyze divergence for a specific time period.

    Args:
        ticker: Stock ticker symbol
        period_name: Name of the period (e.g., "1 Week", "1 Month")
        lookback_days: Number of days for analyst rating lookback

    Returns:
        Complete analysis dict for the period with normalized scores in [-1, 1]
    """
    technical = calculate_technical_score(ticker)
    fundamental = calculate_fundamental_score(ticker, lookback_days)

    # Use normalized scores for divergence detection
    technical_score_normalized = technical["normalized_score"]
    fundamental_score_normalized = fundamental["score"]  # Already in [-1, 1]

    divergence = detect_divergence(technical_score_normalized, fundamental_score_normalized)

    return {
        "period": period_name,
        "lookback_days": lookback_days,
        "technical_analysis": technical,
        "fundamental_analysis": fundamental,
        "technical_score_raw": technical["total_score"],
        "technical_score_normalized": technical_score_normalized,
        "fundamental_score_normalized": fundamental_score_normalized,
        "divergence": divergence
    }


@tool
def analyze_divergence_tool(ticker: str) -> str:
    """
    Analyze divergence between technical and fundamental signals for a stock.

    Calculates technical scores (RSI + Moving Averages) and fundamental scores (Analyst Ratings)
    for three time periods: 1 week, 1 month, and 3 months.

    Technical Score Components:
    - RSI: Oversold (+1), Neutral (0), Overbought (-1)
    - Moving Averages: Bullish Trend (+1), Neutral (0), Bearish Trend (-1)
    - Combined Raw Score: ranges from -2 to +2
    - Normalized Score: z-normalized to range [-1, 1]

    Fundamental Score:
    - Analyst Ratings: Buy/Strong Buy (+1), Hold (0), Sell/Strong Sell (-1)
    - Already normalized in range [-1, 1]

    Divergence Detection (uses normalized scores):
    - Bearish Divergence: Technical bullish but fundamentals bearish
    - Bullish Divergence: Technical bearish but fundamentals bullish
    - Convergence: Both agree (bullish or bearish)

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, NVDA)

    Returns:
        Formatted analysis report with divergence signals for all time periods
    """
    try:
        # Analyze for three time periods
        week_1 = analyze_divergence_for_period(ticker, "1 Week", lookback_days=7)
        month_1 = analyze_divergence_for_period(ticker, "1 Month", lookback_days=30)
        month_3 = analyze_divergence_for_period(ticker, "3 Months", lookback_days=90)

        # Format output
        output = f"=== DIVERGENCE ANALYSIS: {ticker} ===\n\n"

        for analysis in [week_1, month_1, month_3]:
            output += f"--- {analysis['period']} Analysis ---\n\n"

            # Technical Analysis
            output += "TECHNICAL SIGNALS:\n"
            rsi = analysis['technical_analysis']['rsi']
            ma = analysis['technical_analysis']['moving_averages']

            output += f"  RSI: {rsi['value']} ({rsi['condition']}) - Score: {rsi['score']}\n"
            output += f"  Moving Averages: {ma['trend']} - Score: {ma['score']}\n"
            if ma['current_price']:
                output += f"    Current Price: ${ma['current_price']}\n"
                output += f"    50-day MA: ${ma['ma_50']}\n"
                output += f"    200-day MA: ${ma['ma_200']}\n"
            output += f"  TOTAL TECHNICAL SCORE (raw): {analysis['technical_score_raw']}\n"
            output += f"  TOTAL TECHNICAL SCORE (normalized): {analysis['technical_score_normalized']:.2f}\n\n"

            # Fundamental Analysis
            output += "FUNDAMENTAL SIGNALS:\n"
            fund = analysis['fundamental_analysis']
            output += f"  Analyst Rating: {fund['rating_label']}"
            if fund['consensus_rating']:
                output += f" ({fund['consensus_rating']:.2f})"
            output += f" - Score: {fund['score']}\n"

            if fund.get('rating_trend'):
                trend = fund['rating_trend']
                output += f"  Rating Trend: {trend.get('trend_label', 'N/A')}\n"
                output += f"  {trend.get('summary', '')}\n"
            output += f"  TOTAL FUNDAMENTAL SCORE (normalized): {analysis['fundamental_score_normalized']:.2f}\n\n"

            # Divergence Analysis
            div = analysis['divergence']
            output += f"DIVERGENCE SIGNAL: {div['type']}\n"
            output += f"  {div['description']}\n"
            output += f"  ACTION: {div['signal']}\n\n"
            output += "-" * 60 + "\n\n"

        return output

    except Exception as e:
        return f"Error analyzing divergence for {ticker}: {str(e)}"
