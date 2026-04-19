import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import math

def rating_to_label(score):
    """
    Convert Yahoo numeric rating to label.
    1.0 = Strong Buy
    2.0 = Buy
    3.0 = Hold
    4.0 = Sell
    5.0 = Strong Sell
    """

    if score is None:
        return "Unknown"

    if score <= 1.5:
        return "Strong Buy"
    elif score <= 2.5:
        return "Buy"
    elif score <= 3.5:
        return "Hold"
    elif score <= 4.5:
        return "Sell"
    else:
        return "Strong Sell"


def get_recent_changes(upgrades_downgrades, days=30):
    """
    Extract recent upgrades/downgrades.
    """
    if upgrades_downgrades is None or upgrades_downgrades.empty:
        return []

    if not isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
        try:
            pd.to_datetime(upgrades_downgrades.index)
        except:
            return []
        
    cutoff = datetime.now() - timedelta(days=days)
    recent = upgrades_downgrades[
        upgrades_downgrades.index >= cutoff
    ]

    changes = []

    for date, row in recent.iterrows():
        changes.append({
            "date": date.strftime("%Y-%m-%d"),
            "firm": row.get("Firm", "Unknown"),
            "action": row.get("Action", ""),
            "from": row.get("FromGrade", ""),
            "to": row.get("ToGrade", "")
        })

    return changes


def summarize_rating_trend(recent_changes):
    """
    Turn a list of recent rating changes into a simple trend signal.
    """
    if not recent_changes:
        return {
            "trend_label": "No Recent Change",
            "summary": "No analyst upgrades or downgrades in the selected lookback window.",
        }

    upgrades = 0
    downgrades = 0

    for change in recent_changes:
        action = (change.get("action") or "").lower()
        from_grade = (change.get("from") or "").lower()
        to_grade = (change.get("to") or "").lower()

        # Primary signal from explicit action text
        if "upgrade" in action or "upgrad" in action:
            upgrades += 1
        elif "downgrade" in action or "downgrad" in action:
            downgrades += 1
        else:
            # Fallback: infer from From/To if possible
            ladder = ["strong sell", "sell", "hold", "buy", "strong buy"]
            try:
                from_idx = ladder.index(from_grade)
                to_idx = ladder.index(to_grade)
                if to_idx > from_idx:
                    upgrades += 1
                elif to_idx < from_idx:
                    downgrades += 1
            except ValueError:
                # Unknown labels; skip
                continue

    net = upgrades - downgrades

    if net >= 2:
        trend_label = "Bullish Shift"
    elif net <= -2:
        trend_label = "Bearish Shift"
    elif net == 1:
        trend_label = "Mild Bullish Shift"
    elif net == -1:
        trend_label = "Mild Bearish Shift"
    else:
        trend_label = "Stable / Mixed"

    summary = (
        f"{upgrades} upgrade(s) vs {downgrades} downgrade(s) in the lookback window."
    )

    return {
        "trend_label": trend_label,
        "summary": summary,
        "upgrades": upgrades,
        "downgrades": downgrades,
        "net_upgrades": net,
    }


def load_analyst_ratings(ticker):
    """
    Main function to load analyst ratings.
    """

    stock = yf.Ticker(ticker)

    info = stock.info
    recs = stock.recommendations

    try:
        consensus = info.get("recommendationMean")
        num_analysts = info.get("numberOfAnalystOpinions")
        upgrades_downgrades = stock.get_upgrades_downgrades()

        recent_changes = get_recent_changes(upgrades_downgrades)
        trend = summarize_rating_trend(recent_changes)

        data = {
            "ticker": ticker,
            "consensus_rating": consensus,
            "rating_label": rating_to_label(consensus),
            "price_target_avg": info.get("targetMeanPrice"),
            "price_target_high": info.get("targetHighPrice"),
            "price_target_low": info.get("targetLowPrice"),
            "num_analysts": num_analysts,
            "recent_changes": recent_changes,
            "rating_trend": trend,
        }

        return data

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }
def get_normalized_fundamental_score(ticker: str) -> dict:
    """
    Returns a normalized fundamental score (0.0-1.0) based on analyst consensus.
    1.0 = Strong Buy, 0.5 = Hold, 0.0 = Strong Sell.
    """
    data = load_analyst_ratings(ticker)
    if "error" in data or data.get("consensus_rating") is None:
        return {"score": 0.5, "confidence": 0.0, "details": data}
    
    rating = data["consensus_rating"]
    # Map [1.0, 5.0] to [1.0, 0.0]
    normalized = (5.0 - rating) / 4.0
    normalized = max(0.0, min(1.0, normalized))
    
    # NEW: Non-linear confidence curve
    # 1 - exp(-N/10) means conviction hits 63% at 10 analysts, 86% at 20.
    num_analysts = data.get("num_analysts", 0) or 0
    confidence = 1 - math.exp(-num_analysts / 10.0)
    
    return {
        "score": round(normalized, 3),
        "confidence": round(confidence, 3),
        "raw_value": rating,
        "label": data.get("rating_label", "Hold"),
        "num_analysts": num_analysts
    }
