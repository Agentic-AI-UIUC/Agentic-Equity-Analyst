"""Command-line tool for calculating stock moving averages."""

from __future__ import annotations

import argparse
import yfinance as yf
from datetime import date, timedelta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate moving average for a stock over a specified time period.",
    )
    parser.add_argument("--company", required=True, help="Company name to analyze.")
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker symbol (e.g., AAPL, MSFT).",
    )
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Number of days for moving average calculation (e.g., 7, 50, 200, 1000).",
    )
    return parser.parse_args()


def calculate_moving_average(ticker: str, days: int) -> float | None:
    """
    Calculate the moving average for a given ticker over the specified number of days.

    Args:
        ticker: Stock ticker symbol
        days: Number of days for the moving average

    Returns:
        The moving average value or None if data is insufficient
    """
    try:
        # Fetch data with some buffer to ensure we have enough data points
        stock = yf.Ticker(ticker)
        end_date = date.today()
        # Add extra days to account for weekends/holidays
        start_date = end_date - timedelta(days=days + 100)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            print(f"Error: No data found for ticker {ticker}")
            return None

        if len(hist) < days:
            print(f"Warning: Only {len(hist)} days of data available, requested {days} days")
            # Use whatever data is available
            days = len(hist)

        # Calculate moving average using the most recent 'days' closing prices
        moving_avg = hist['Close'].tail(days).mean()

        return moving_avg

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def main() -> None:
    args = parse_args()

    print(f"Calculating {args.days}-day moving average for {args.company} ({args.ticker})...")

    moving_avg = calculate_moving_average(args.ticker, args.days)

    if moving_avg is not None:
        print(f"\nMoving average for period of last {args.days} days: ${moving_avg:.2f}")
    else:
        print(f"\nFailed to calculate moving average for {args.company} ({args.ticker})")


if __name__ == "__main__":
    main()
