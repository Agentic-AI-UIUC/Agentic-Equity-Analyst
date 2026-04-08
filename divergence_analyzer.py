import yfinance as yf
from datetime import date, timedelta
from langchain.tools import tool
from analyst_ratings_loader import load_analyst_ratings
import numpy as np


def calculate_price_trend(ticker: str, lookback_days: int = 30):
    """
    Calculate price trend (slope) over a lookback period.

    Returns:
        dict with slope, direction, and interpretation
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < 5:
            return {
                "slope": 0,
                "direction": "INSUFFICIENT_DATA",
                "price_change_pct": 0,
                "start_price": None,
                "end_price": None
            }

        # Get the last N trading days
        recent_hist = hist.tail(min(lookback_days, len(hist)))
        prices = recent_hist['Close'].values

        if len(prices) < 2:
            return {
                "slope": 0,
                "direction": "INSUFFICIENT_DATA",
                "price_change_pct": 0,
                "start_price": None,
                "end_price": None
            }

        # Calculate linear regression slope
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)

        # Calculate percentage change
        start_price = prices[0]
        end_price = prices[-1]
        price_change_pct = ((end_price - start_price) / start_price) * 100

        # Determine direction based on slope
        # Normalize slope relative to price level
        normalized_slope = (slope / start_price) * 100  # slope as % per day

        if normalized_slope > 0.1:  # Rising more than 0.1% per day
            direction = "RISING"
        elif normalized_slope < -0.1:  # Falling more than 0.1% per day
            direction = "FALLING"
        else:
            direction = "FLAT"

        return {
            "slope": slope,
            "normalized_slope": normalized_slope,
            "direction": direction,
            "price_change_pct": price_change_pct,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2)
        }

    except Exception as e:
        return {
            "slope": 0,
            "direction": f"ERROR: {str(e)}",
            "price_change_pct": 0,
            "start_price": None,
            "end_price": None
        }


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


def detect_divergence(technical_score: float, fundamental_score: float, divergence_value: float, price_trend: dict):
    """
    Detect divergence between technical and fundamental signals, incorporating price direction.

    Classical Divergences:
    - Bullish Divergence: Price falling/flat + Technical signals rising
    - Bearish Divergence: Price rising + Technical signals falling

    Args:
        technical_score: Normalized technical score in range [-1, 1]
        fundamental_score: Normalized fundamental score in range [-1, 1]
        divergence_value: T_norm - F_norm (positive = technical stronger, negative = fundamental stronger)
        price_trend: Dict with price direction (RISING, FALLING, FLAT) and slope

    Returns:
        dict with divergence type, value, strength indicator, and interpretation
    """
    # Calculate combined score (50% technical, 50% fundamental)
    combined_score = 0.5 * technical_score + 0.5 * fundamental_score

    # Determine which signal is stronger
    abs_divergence = abs(divergence_value)

    # Detect THRESHOLD-BASED divergence when Technical >= 0.4 AND Fundamental <= -0.3
    threshold_divergence_bullish_tech = (technical_score >= 0.4 and fundamental_score <= -0.3)
    # Also check opposite: Technical <= -0.4 AND Fundamental >= 0.3
    threshold_divergence_bearish_tech = (technical_score <= -0.4 and fundamental_score >= 0.3)

    # Detect STRONG divergence when |D| > 1
    is_strong = abs_divergence > 1.0

    if abs_divergence < 0.15:
        strength_comparison = "ALIGNED - Technical and fundamental signals are approximately equal"
    elif divergence_value > 0:
        strength_comparison = f"TECHNICAL DOMINANT - Technical signals are stronger than fundamentals by {abs_divergence:.2f}"
    else:
        strength_comparison = f"FUNDAMENTAL DOMINANT - Fundamental signals are stronger than technicals by {abs_divergence:.2f}"

    # Get price direction
    price_direction = price_trend.get("direction", "UNKNOWN")

    # Detect classical divergences based on price vs technical signals
    classical_divergence = None
    if price_direction in ["FALLING", "FLAT"] and technical_score > 0:
        # Price falling/flat but technicals bullish = Classical Bullish Divergence
        classical_divergence = "CLASSICAL_BULLISH_DIVERGENCE"
    elif price_direction == "RISING" and technical_score < 0:
        # Price rising but technicals bearish = Classical Bearish Divergence
        classical_divergence = "CLASSICAL_BEARISH_DIVERGENCE"

    # Determine divergence type based on sign agreement and classical patterns
    # PRIORITY: Check threshold-based divergence first (Technical >= 0.4, Fundamental <= -0.3)
    if threshold_divergence_bullish_tech:
        # THRESHOLD DIVERGENCE DETECTED: Bullish technicals, bearish fundamentals
        div_type = "THRESHOLD_DIVERGENCE_DETECTED"
        if classical_divergence == "CLASSICAL_BULLISH_DIVERGENCE":
            description = f"🚨 DIVERGENCE DETECTED (Threshold-Based): Technical score {technical_score:.2f} >= 0.4 AND Fundamental score {fundamental_score:.2f} <= -0.3. Price {price_direction.lower()} with bullish technicals but bearish fundamentals. Classical bullish divergence present - strong reversal signal."
            signal = "HIGH OPPORTUNITY - Threshold divergence with classical bullish pattern"
        else:
            description = f"🚨 DIVERGENCE DETECTED (Threshold-Based): Technical score {technical_score:.2f} >= 0.4 AND Fundamental score {fundamental_score:.2f} <= -0.3. Price {price_direction.lower()}. Technicals bullish but fundamentals bearish - significant disconnect."
            signal = "CAUTION - Threshold divergence suggests technical strength not supported by fundamentals"

    elif technical_score > 0 and fundamental_score < 0:
        # Technical bullish, Fundamental bearish (but not meeting threshold)
        if classical_divergence == "CLASSICAL_BULLISH_DIVERGENCE":
            div_type = "STRONG_BULLISH_DIVERGENCE" if is_strong else "BULLISH_DIVERGENCE"
            if is_strong:
                description = f"STRONG CLASSICAL BULLISH DIVERGENCE: Price is {price_direction.lower()} but technical indicators are very bullish. Fundamentals bearish. Strong reversal signal - price may be bottoming."
                signal = "HIGH OPPORTUNITY - Classical divergence with strong technical bounce potential"
            else:
                description = f"CLASSICAL BULLISH DIVERGENCE: Price is {price_direction.lower()} but technical indicators are bullish. Fundamentals bearish. Potential reversal or stabilization."
                signal = "OPPORTUNITY - Watch for price reversal despite bearish fundamentals"
        else:
            div_type = "STRONG_BEARISH_DIVERGENCE" if is_strong else "BEARISH_DIVERGENCE"
            if is_strong:
                description = f"STRONG DIVERGENCE: Technical indicators are very bullish (price {price_direction.lower()}) while fundamentals are bearish. Significant disconnect - price may be overextended."
                signal = "HIGH CAUTION - Major divergence suggests elevated risk"
            else:
                description = f"Technical indicators are bullish (price {price_direction.lower()}) but fundamentals are bearish. Price may be overextended."
                signal = "CAUTION - Technical strength not supported by fundamentals"

    elif threshold_divergence_bearish_tech:
        # THRESHOLD DIVERGENCE DETECTED: Bearish technicals, bullish fundamentals
        div_type = "THRESHOLD_DIVERGENCE_DETECTED"
        if classical_divergence == "CLASSICAL_BEARISH_DIVERGENCE":
            description = f"🚨 DIVERGENCE DETECTED (Threshold-Based): Technical score {technical_score:.2f} <= -0.4 AND Fundamental score {fundamental_score:.2f} >= 0.3. Price {price_direction.lower()} with bearish technicals but bullish fundamentals. Classical bearish divergence present - potential downturn signal."
            signal = "HIGH CAUTION - Threshold divergence with classical bearish pattern"
        else:
            description = f"🚨 DIVERGENCE DETECTED (Threshold-Based): Technical score {technical_score:.2f} <= -0.4 AND Fundamental score {fundamental_score:.2f} >= 0.3. Price {price_direction.lower()}. Technicals bearish but fundamentals bullish - significant disconnect, potential opportunity."
            signal = "OPPORTUNITY - Threshold divergence suggests fundamentals not reflected in technicals"

    elif technical_score < 0 and fundamental_score > 0:
        # Technical bearish, Fundamental bullish (but not meeting threshold)
        if classical_divergence == "CLASSICAL_BEARISH_DIVERGENCE":
            div_type = "STRONG_BEARISH_DIVERGENCE" if is_strong else "BEARISH_DIVERGENCE"
            if is_strong:
                description = f"STRONG CLASSICAL BEARISH DIVERGENCE: Price is {price_direction.lower()} but technical indicators are very bearish. Fundamentals bullish. Strong reversal warning - uptrend may be exhausting."
                signal = "HIGH CAUTION - Classical divergence suggests potential downturn"
            else:
                description = f"CLASSICAL BEARISH DIVERGENCE: Price is {price_direction.lower()} but technical indicators are bearish. Fundamentals bullish. Potential reversal or pullback."
                signal = "CAUTION - Watch for price weakness despite bullish fundamentals"
        else:
            div_type = "STRONG_BULLISH_DIVERGENCE" if is_strong else "BULLISH_DIVERGENCE"
            if is_strong:
                description = f"STRONG DIVERGENCE: Technical indicators are very bearish (price {price_direction.lower()}) while fundamentals are bullish. Significant disconnect - price may be severely undervalued."
                signal = "HIGH OPPORTUNITY - Major divergence suggests strong upside potential"
            else:
                description = f"Technical indicators are bearish (price {price_direction.lower()}) but fundamentals are bullish. Price may be oversold relative to value."
                signal = "OPPORTUNITY - Fundamental strength not reflected in technicals"

    elif technical_score > 0 and fundamental_score > 0:
        # Both bullish
        if classical_divergence == "CLASSICAL_BULLISH_DIVERGENCE":
            div_type = "STRONG_BULLISH_CONVERGENCE" if is_strong else "BULLISH_CONVERGENCE"
            if is_strong:
                description = f"STRONG CLASSICAL BULLISH SETUP: Price {price_direction.lower()} but both technical and fundamental indicators are very bullish. Exceptional reversal/continuation signal."
                signal = "VERY STRONG BUY - Classical bullish divergence with fundamental support"
            else:
                description = f"CLASSICAL BULLISH SETUP: Price {price_direction.lower()} but both technical and fundamental indicators are bullish. Strong reversal potential."
                signal = "STRONG BUY - Classical divergence with fundamental alignment"
        else:
            div_type = "STRONG_BULLISH_CONVERGENCE" if is_strong else "BULLISH_CONVERGENCE"
            if is_strong:
                description = f"STRONG CONVERGENCE: Price {price_direction.lower()} and both technical and fundamental indicators are very bullish. Exceptional buy signal."
                signal = "VERY STRONG BUY - Powerful alignment across all signals"
            else:
                description = f"Price {price_direction.lower()} and both technical and fundamental indicators are bullish. Strong buy signal."
                signal = "STRONG BUY - Technical and fundamental alignment"

    elif technical_score < 0 and fundamental_score < 0:
        # Both bearish
        if classical_divergence == "CLASSICAL_BEARISH_DIVERGENCE":
            div_type = "STRONG_BEARISH_CONVERGENCE" if is_strong else "BEARISH_CONVERGENCE"
            if is_strong:
                description = f"STRONG CLASSICAL BEARISH SETUP: Price {price_direction.lower()} but both technical and fundamental indicators are very bearish. Exceptional reversal/continuation warning."
                signal = "VERY STRONG SELL - Classical bearish divergence with fundamental weakness"
            else:
                description = f"CLASSICAL BEARISH SETUP: Price {price_direction.lower()} but both technical and fundamental indicators are bearish. Strong reversal risk."
                signal = "STRONG SELL - Classical divergence with fundamental weakness"
        else:
            div_type = "STRONG_BEARISH_CONVERGENCE" if is_strong else "BEARISH_CONVERGENCE"
            if is_strong:
                description = f"STRONG CONVERGENCE: Price {price_direction.lower()} and both technical and fundamental indicators are very bearish. Exceptional sell signal."
                signal = "VERY STRONG SELL - Powerful alignment across all signals"
            else:
                description = f"Price {price_direction.lower()} and both technical and fundamental indicators are bearish. Strong sell signal."
                signal = "STRONG SELL - Technical and fundamental alignment"

    else:
        div_type = "NEUTRAL_MIXED"
        description = f"Price {price_direction.lower()} with mixed or neutral signals from technical and fundamental indicators."
        signal = "HOLD - Wait for clearer signals"

    return {
        "type": div_type,
        "is_strong_divergence": is_strong,
        "threshold_divergence_detected": threshold_divergence_bullish_tech or threshold_divergence_bearish_tech,
        "classical_divergence": classical_divergence,
        "divergence_value": divergence_value,
        "combined_score": combined_score,
        "strength_comparison": strength_comparison,
        "description": description,
        "signal": signal
    }


def analyze_divergence_for_period(ticker: str, period_name: str, lookback_days: int):
    """
    Analyze divergence for a specific time period.

    Args:
        ticker: Stock ticker symbol
        period_name: Name of the period (e.g., "1 Week", "1 Month")
        lookback_days: Number of days for analyst rating lookback

    Returns:
        Complete analysis dict for the period with normalized scores, price trend, and divergence value
    """
    technical = calculate_technical_score(ticker)
    fundamental = calculate_fundamental_score(ticker, lookback_days)
    price_trend = calculate_price_trend(ticker, lookback_days)

    # Use normalized scores for divergence detection
    technical_score_normalized = technical["normalized_score"]
    fundamental_score_normalized = fundamental["score"]  # Already in [-1, 1]

    # Calculate divergence: T_norm(t) - F_norm(t)
    # Positive = technical stronger, Negative = fundamental stronger, ~0 = aligned
    divergence_value = technical_score_normalized - fundamental_score_normalized

    divergence = detect_divergence(
        technical_score_normalized,
        fundamental_score_normalized,
        divergence_value,
        price_trend
    )

    return {
        "period": period_name,
        "lookback_days": lookback_days,
        "technical_analysis": technical,
        "fundamental_analysis": fundamental,
        "price_trend": price_trend,
        "technical_score_raw": technical["total_score"],
        "technical_score_normalized": technical_score_normalized,
        "fundamental_score_normalized": fundamental_score_normalized,
        "divergence_value": divergence_value,
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
    - Normalized Score (T_norm): z-normalized to range [-1, 1]

    Fundamental Score (F_norm):
    - Analyst Ratings: Buy/Strong Buy (+1), Hold (0), Sell/Strong Sell (-1)
    - Already normalized in range [-1, 1]

    Combined Score Calculation (50/50 Weighting):
    - Combined Score = 0.5 × T_norm + 0.5 × F_norm
    - Ranges from -1 (very bearish) to +1 (very bullish)
    - Represents overall market sentiment equally weighting technicals and fundamentals

    Divergence Calculation:
    - Divergence Value = T_norm(t) - F_norm(t) for each time period t
    - Positive value: Technical signals are stronger than fundamentals
    - Negative value: Fundamental signals are stronger than technicals
    - Near zero (|value| < 0.15): Technical and fundamental signals are aligned

    Threshold-Based Divergence Detection (Primary):
    - 🚨 DIVERGENCE DETECTED when:
      → Technical >= 0.4 AND Fundamental <= -0.3 (Bullish tech, bearish fundamental)
      → Technical <= -0.4 AND Fundamental >= 0.3 (Bearish tech, bullish fundamental)
    - This identifies significant disconnects between technical strength and fundamental weakness (or vice versa)

    Price Trend Analysis:
    - Calculates price direction (RISING, FALLING, FLAT) using linear regression slope
    - Measures percentage change over the period
    - Direction determines classical divergence patterns

    Classical Divergence Detection (Price vs Technical Signals):
    - Classical Bullish Divergence: Price falling/flat BUT technicals bullish
      → Indicates potential price reversal upward (price may be bottoming)
    - Classical Bearish Divergence: Price rising BUT technicals bearish
      → Indicates potential price reversal downward (uptrend may be exhausting)

    Strong Divergence Detection:
    - STRONG divergence when |D| > 1.0
    - Indicates extreme disconnect between technical and fundamental signals
    - Strong Bullish Divergence: Technicals very bearish, fundamentals bullish (high opportunity)
    - Strong Bearish Divergence: Technicals very bullish, fundamentals bearish (high caution)
    - Strong Convergence: Both very bullish or very bearish (exceptional signal strength)

    Regular Divergence Detection (uses normalized scores):
    - Bearish Divergence: Technical bullish but fundamentals bearish
    - Bullish Divergence: Technical bearish but fundamentals bullish
    - Convergence: Both agree (bullish or bearish)

    Combined Analysis:
    - Integrates price direction, technical signals, and fundamental sentiment
    - Identifies when price action contradicts technical indicators (classical divergence)
    - Measures strength of signal disagreement (|D| value)
    - Provides actionable trading signals based on all three dimensions

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, NVDA)

    Returns:
        Formatted analysis report with price trends, divergence values, classical patterns, and signals for all time periods
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

            # Price Trend Analysis
            price_trend = analysis['price_trend']
            output += "PRICE TREND:\n"
            output += f"  Direction: {price_trend['direction']}\n"
            if price_trend.get('start_price') and price_trend.get('end_price'):
                output += f"  Price Change: ${price_trend['start_price']} → ${price_trend['end_price']} ({price_trend['price_change_pct']:+.2f}%)\n"
            if 'normalized_slope' in price_trend:
                output += f"  Trend Slope: {price_trend['normalized_slope']:.4f}% per day\n"
            output += "\n"

            # Divergence Analysis
            div = analysis['divergence']
            div_val = analysis['divergence_value']
            is_strong = div['is_strong_divergence']
            classical_div = div.get('classical_divergence')
            threshold_div = div.get('threshold_divergence_detected', False)
            combined = div.get('combined_score', 0)

            # Highlight threshold, strong, or classical divergences
            if threshold_div or is_strong or classical_div:
                output += "=" * 60 + "\n"
                if threshold_div:
                    output += "*** 🚨 THRESHOLD DIVERGENCE DETECTED 🚨 ***\n"
                if is_strong:
                    output += "*** STRONG DIVERGENCE DETECTED (|D| > 1) ***\n"
                if classical_div:
                    output += f"*** {classical_div.replace('_', ' ')} DETECTED ***\n"
                output += "=" * 60 + "\n\n"

            output += f"DIVERGENCE ANALYSIS:\n"
            output += f"  Combined Score (50% Tech + 50% Fund): {combined:+.2f}\n"
            output += f"  Divergence Value (T_norm - F_norm): {div_val:+.2f}\n"
            if threshold_div:
                output += f"  🚨 THRESHOLD CRITERIA MET:\n"
                if analysis['technical_score_normalized'] >= 0.4:
                    output += f"     Technical: {analysis['technical_score_normalized']:.2f} >= 0.4 ✓\n"
                if analysis['fundamental_score_normalized'] <= -0.3:
                    output += f"     Fundamental: {analysis['fundamental_score_normalized']:.2f} <= -0.3 ✓\n"
                if analysis['technical_score_normalized'] <= -0.4:
                    output += f"     Technical: {analysis['technical_score_normalized']:.2f} <= -0.4 ✓\n"
                if analysis['fundamental_score_normalized'] >= 0.3:
                    output += f"     Fundamental: {analysis['fundamental_score_normalized']:.2f} >= 0.3 ✓\n"
            if is_strong:
                output += f"  Magnitude: |{div_val:.2f}| > 1.0 → STRONG DIVERGENCE\n"
            if classical_div:
                if classical_div == "CLASSICAL_BULLISH_DIVERGENCE":
                    output += f"  Pattern: Price {price_trend['direction'].lower()} + Technicals bullish → Classical Bullish Divergence\n"
                elif classical_div == "CLASSICAL_BEARISH_DIVERGENCE":
                    output += f"  Pattern: Price {price_trend['direction'].lower()} + Technicals bearish → Classical Bearish Divergence\n"
            output += f"  {div['strength_comparison']}\n\n"
            output += f"DIVERGENCE SIGNAL: {div['type']}\n"
            output += f"  {div['description']}\n"
            output += f"  ACTION: {div['signal']}\n\n"

            if threshold_div or is_strong or classical_div:
                output += "=" * 60 + "\n\n"
            else:
                output += "-" * 60 + "\n\n"

        return output

    except Exception as e:
        return f"Error analyzing divergence for {ticker}: {str(e)}"
