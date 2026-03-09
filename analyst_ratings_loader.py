import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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
            #now = pd.Timestamp.now()
            #upgrades_downgrades.index = upgrades_downgrades.index.map(lambda i: now - pd.DateOffset(months=int(i)))
            pd.to_datetime(upgrades_downgrades.index)
        except:
            return []
        
    cutoff = datetime.now() - timedelta(days=days)
    recent = upgrades_downgrades[
        upgrades_downgrades.index >= cutoff
    ]

    changes = []

    # Need to implement
    for date, row in recent.iterrows():
        changes.append({
            "date": date.strftime("%Y-%m-%d"),
            "firm": row.get("Firm", "Unknown"),
            "action": row.get("Action", ""),
            "from": row.get("FromGrade", ""),
            "to": row.get("ToGrade", "")
        })

    return changes


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

        data = {
            "ticker": ticker,
            "consensus_rating": consensus,
            "rating_label": rating_to_label(consensus),
            "price_target_avg": info.get("targetMeanPrice"),
            "price_target_high": info.get("targetHighPrice"),
            "price_target_low": info.get("targetLowPrice"),
            "num_analysts": num_analysts,
            "recent_changes": get_recent_changes(upgrades_downgrades)
        }

        return data

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }
    
data = load_analyst_ratings("NFLX")

print(data)