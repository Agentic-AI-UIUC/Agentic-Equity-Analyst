"""Market regime detection using Kalshi prediction-market data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kalshi_client import KalshiClient, KalshiAPIError

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────

class Regime(Enum):
    """Discretised macro regime buckets."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class RegimeSignal:
    """One signal contributing to regime determination."""

    name: str
    kalshi_ticker: str
    implied_probability: float  # 0.0 – 1.0
    signal_direction: str       # "bullish" | "bearish" | "neutral"
    weight: float               # 0.0 – 1.0
    explanation: str


@dataclass
class RegimeAssessment:
    """Final regime output with all supporting signals."""

    regime: Regime
    confidence: float           # 0.0 – 1.0
    signals: list[RegimeSignal] = field(default_factory=list)
    summary: str = ""


# ── Helpers ──────────────────────────────────────────────────────────

def _price_to_prob(market: dict[str, Any]) -> float:
    """Extract the best available probability from a market dict.

    Preference order: last_price > yes_bid > yes_ask (midpoint).
    All dollar-denominated string fields are in [0, 1].
    """
    for key in ("last_price", "last_price_dollars"):
        val = market.get(key)
        if val is not None:
            return float(val)
    bid = market.get("yes_bid", market.get("yes_bid_dollars"))
    ask = market.get("yes_ask", market.get("yes_ask_dollars"))
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return 0.5  # fallback: maximally uncertain


def _find_best_market(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the open market with the highest volume, or None."""
    open_markets = [m for m in markets if m.get("status") == "open"]
    if not open_markets:
        open_markets = markets  # fallback to all
    if not open_markets:
        return None
    return max(open_markets, key=lambda m: float(m.get("volume", m.get("volume_fp", 0))))


def _search_series_markets(
    client: KalshiClient,
    series_ticker: str,
    status: str = "open",
) -> list[dict[str, Any]]:
    """Fetch markets for a given series ticker, returning an empty list on failure."""
    try:
        data = client.get_markets(series_ticker=series_ticker, status=status, limit=50)
        return data.get("markets", [])
    except (KalshiAPIError, Exception) as exc:
        logger.warning("Could not fetch markets for series %s: %s", series_ticker, exc)
        return []


# ── Signal builders ──────────────────────────────────────────────────

def _recession_signal(client: KalshiClient) -> RegimeSignal | None:
    """Recession probability signal (weight 0.30)."""
    series_candidates = ["KXRECESSION", "KXRECSSNBER", "KXRECESSNBER"]
    for series in series_candidates:
        markets = _search_series_markets(client, series)
        if markets:
            break
    else:
        # Try broader search
        try:
            data = client.get_markets(status="open", limit=100)
            markets = [
                m for m in data.get("markets", [])
                if "recession" in (m.get("title", "") or "").lower()
            ]
        except Exception:
            markets = []

    if not markets:
        logger.info("No recession markets found on Kalshi.")
        return None

    best = _find_best_market(markets)
    if best is None:
        return None

    prob = _price_to_prob(best)
    if prob > 0.40:
        direction = "bearish"
    elif prob > 0.15:
        direction = "neutral"
    else:
        direction = "bullish"

    return RegimeSignal(
        name="Recession Probability",
        kalshi_ticker=best.get("ticker", ""),
        implied_probability=prob,
        signal_direction=direction,
        weight=0.30,
        explanation=f"Market implies {prob:.0%} chance of recession.",
    )


def _sp500_signal(client: KalshiClient) -> RegimeSignal | None:
    """S&P 500 directional signal (weight 0.25)."""
    series_candidates = ["KXSP500", "KXSP500Y", "KXSP500ADDQ"]
    for series in series_candidates:
        markets = _search_series_markets(client, series)
        if markets:
            break
    else:
        try:
            data = client.get_markets(status="open", limit=200)
            markets = [
                m for m in data.get("markets", [])
                if any(kw in (m.get("title", "") or "").lower() for kw in ("s&p 500", "sp500", "s&p500"))
            ]
        except Exception:
            markets = []

    if not markets:
        logger.info("No S&P 500 markets found on Kalshi.")
        return None

    # Look for upside-leaning vs downside-leaning markets.
    # Heuristic: markets with "above" / "higher" / "up" vs "below" / "lower" / "down".
    bullish_kws = {"above", "higher", "up", "rise", "gain", "added"}
    bearish_kws = {"below", "lower", "down", "fall", "drop", "removed"}

    bull_prob_sum, bull_count = 0.0, 0
    bear_prob_sum, bear_count = 0.0, 0

    for m in markets:
        title = (m.get("title") or "").lower()
        prob = _price_to_prob(m)
        if any(kw in title for kw in bullish_kws):
            bull_prob_sum += prob
            bull_count += 1
        elif any(kw in title for kw in bearish_kws):
            bear_prob_sum += prob
            bear_count += 1

    # If no keyword match, use the single best market
    if bull_count == 0 and bear_count == 0:
        best = _find_best_market(markets)
        if best is None:
            return None
        prob = _price_to_prob(best)
        direction = "bullish" if prob > 0.55 else ("bearish" if prob < 0.45 else "neutral")
        return RegimeSignal(
            name="S&P 500 Outlook",
            kalshi_ticker=best.get("ticker", ""),
            implied_probability=prob,
            signal_direction=direction,
            weight=0.25,
            explanation=f"S&P 500 market priced at {prob:.0%}.",
        )

    bull_avg = bull_prob_sum / bull_count if bull_count else 0.0
    bear_avg = bear_prob_sum / bear_count if bear_count else 0.0
    net = bull_avg - bear_avg  # positive = bullish

    direction = "bullish" if net > 0.10 else ("bearish" if net < -0.10 else "neutral")
    prob = (net + 1.0) / 2.0  # normalise to 0-1 for display

    return RegimeSignal(
        name="S&P 500 Outlook",
        kalshi_ticker=markets[0].get("ticker", ""),
        implied_probability=round(prob, 2),
        signal_direction=direction,
        weight=0.25,
        explanation=f"Bullish avg {bull_avg:.0%} vs bearish avg {bear_avg:.0%}.",
    )


def _fed_rate_signal(client: KalshiClient) -> RegimeSignal | None:
    """Fed interest-rate decision signal (weight 0.15)."""
    series_candidates = ["KXFEDDECISION", "KXFEDRATE", "KXFED"]
    for series in series_candidates:
        markets = _search_series_markets(client, series)
        if markets:
            break
    else:
        try:
            data = client.get_markets(status="open", limit=200)
            markets = [
                m for m in data.get("markets", [])
                if any(kw in (m.get("title", "") or "").lower() for kw in ("fed", "fomc", "interest rate"))
            ]
        except Exception:
            markets = []

    if not markets:
        logger.info("No Fed rate markets found on Kalshi.")
        return None

    # Look for cut / hike language
    cut_kws = {"cut", "lower", "decrease", "below"}
    hike_kws = {"hike", "raise", "increase", "above", "higher"}
    hold_kws = {"hold", "unchanged", "same", "no change"}

    cut_prob, hike_prob, hold_prob = 0.0, 0.0, 0.0
    cut_n, hike_n, hold_n = 0, 0, 0

    for m in markets:
        title = (m.get("title") or "").lower()
        prob = _price_to_prob(m)
        if any(kw in title for kw in cut_kws):
            cut_prob += prob
            cut_n += 1
        elif any(kw in title for kw in hike_kws):
            hike_prob += prob
            hike_n += 1
        elif any(kw in title for kw in hold_kws):
            hold_prob += prob
            hold_n += 1

    # Interpret: cuts in a growth environment = bullish; hikes = bearish
    cut_avg = cut_prob / cut_n if cut_n else 0.0
    hike_avg = hike_prob / hike_n if hike_n else 0.0

    if hike_avg > 0.40:
        direction = "bearish"
        explanation = f"Markets price {hike_avg:.0%} chance of rate hike — tightening signal."
    elif cut_avg > 0.50:
        direction = "neutral"  # cuts can be bullish or bearish depending on context
        explanation = f"Markets price {cut_avg:.0%} chance of rate cut — accommodative but potentially signaling weakness."
    else:
        direction = "neutral"
        explanation = "Fed rate expectations are balanced."

    best = _find_best_market(markets)
    ticker = best.get("ticker", "") if best else ""
    display_prob = max(cut_avg, hike_avg, hold_prob / hold_n if hold_n else 0.0)

    return RegimeSignal(
        name="Fed Rate Expectations",
        kalshi_ticker=ticker,
        implied_probability=round(display_prob, 2),
        signal_direction=direction,
        weight=0.15,
        explanation=explanation,
    )


def _inflation_signal(client: KalshiClient) -> RegimeSignal | None:
    """CPI / Inflation signal (weight 0.10)."""
    series_candidates = ["KXCPI", "KXCPIY", "KXINFLATION"]
    for series in series_candidates:
        markets = _search_series_markets(client, series)
        if markets:
            break
    else:
        try:
            data = client.get_markets(status="open", limit=200)
            markets = [
                m for m in data.get("markets", [])
                if any(kw in (m.get("title", "") or "").lower() for kw in ("cpi", "inflation"))
            ]
        except Exception:
            markets = []

    if not markets:
        logger.info("No CPI/inflation markets found on Kalshi.")
        return None

    # High-inflation brackets being priced up = bearish
    hot_kws = {"above", "over", "higher", "exceed"}
    cool_kws = {"below", "under", "lower"}

    hot_prob_sum, hot_n = 0.0, 0
    cool_prob_sum, cool_n = 0.0, 0

    for m in markets:
        title = (m.get("title") or "").lower()
        prob = _price_to_prob(m)
        if any(kw in title for kw in hot_kws):
            hot_prob_sum += prob
            hot_n += 1
        elif any(kw in title for kw in cool_kws):
            cool_prob_sum += prob
            cool_n += 1

    hot_avg = hot_prob_sum / hot_n if hot_n else 0.0
    cool_avg = cool_prob_sum / cool_n if cool_n else 0.0

    if hot_avg > 0.50:
        direction = "bearish"
        explanation = f"Inflation markets lean hot ({hot_avg:.0%} avg on above-target brackets)."
    elif cool_avg > 0.50:
        direction = "bullish"
        explanation = f"Inflation cooling — below-target brackets at {cool_avg:.0%}."
    else:
        direction = "neutral"
        explanation = "Inflation expectations roughly balanced."

    best = _find_best_market(markets)
    ticker = best.get("ticker", "") if best else ""
    display_prob = max(hot_avg, cool_avg) if (hot_n or cool_n) else _price_to_prob(best) if best else 0.5

    return RegimeSignal(
        name="Inflation / CPI",
        kalshi_ticker=ticker,
        implied_probability=round(display_prob, 2),
        signal_direction=direction,
        weight=0.10,
        explanation=explanation,
    )


def _gdp_signal(client: KalshiClient) -> RegimeSignal | None:
    """GDP growth signal (weight 0.20)."""
    series_candidates = ["KXGDP", "KXGDPY", "KXGDPYEAR", "KXGDPQ"]
    for series in series_candidates:
        markets = _search_series_markets(client, series)
        if markets:
            break
    else:
        try:
            data = client.get_markets(status="open", limit=200)
            markets = [
                m for m in data.get("markets", [])
                if "gdp" in (m.get("title", "") or "").lower()
            ]
        except Exception:
            markets = []

    if not markets:
        logger.info("No GDP markets found on Kalshi.")
        return None

    # High-growth brackets priced up = bullish
    high_kws = {"above", "over", "higher", "exceed", "growth"}
    low_kws = {"below", "under", "negative", "contraction"}

    high_sum, high_n = 0.0, 0
    low_sum, low_n = 0.0, 0

    for m in markets:
        title = (m.get("title") or "").lower()
        prob = _price_to_prob(m)
        if any(kw in title for kw in high_kws):
            high_sum += prob
            high_n += 1
        elif any(kw in title for kw in low_kws):
            low_sum += prob
            low_n += 1

    high_avg = high_sum / high_n if high_n else 0.0
    low_avg = low_sum / low_n if low_n else 0.0

    if high_avg > 0.50:
        direction = "bullish"
        explanation = f"GDP growth markets favour upside ({high_avg:.0%} avg)."
    elif low_avg > 0.50:
        direction = "bearish"
        explanation = f"GDP markets lean toward contraction/low growth ({low_avg:.0%} avg)."
    else:
        direction = "neutral"
        explanation = "GDP growth expectations are mixed."

    best = _find_best_market(markets)
    ticker = best.get("ticker", "") if best else ""
    display_prob = max(high_avg, low_avg) if (high_n or low_n) else _price_to_prob(best) if best else 0.5

    return RegimeSignal(
        name="GDP Growth",
        kalshi_ticker=ticker,
        implied_probability=round(display_prob, 2),
        signal_direction=direction,
        weight=0.20,
        explanation=explanation,
    )


# ── Regime scorer ────────────────────────────────────────────────────

_DIRECTION_SCORE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def _score_to_regime(score: float) -> Regime:
    """Map a weighted score in [-1, 1] to a Regime enum value."""
    if score >= 0.45:
        return Regime.STRONG_BULL
    if score >= 0.15:
        return Regime.BULL
    if score > -0.15:
        return Regime.NEUTRAL
    if score > -0.45:
        return Regime.BEAR
    return Regime.STRONG_BEAR


_REGIME_LABELS = {
    Regime.STRONG_BULL: "strongly bullish",
    Regime.BULL: "moderately bullish",
    Regime.NEUTRAL: "neutral",
    Regime.BEAR: "moderately bearish",
    Regime.STRONG_BEAR: "strongly bearish",
}


def assess_regime(client: KalshiClient | None = None) -> RegimeAssessment:
    """Fetch data from multiple Kalshi markets and compute a weighted regime score.

    Parameters
    ----------
    client : KalshiClient, optional
        An existing client instance.  A default one is created if omitted.

    Returns
    -------
    RegimeAssessment
        The overall regime with supporting signals.
    """
    if client is None:
        client = KalshiClient()

    signal_builders = [
        _recession_signal,
        _sp500_signal,
        _gdp_signal,
        _fed_rate_signal,
        _inflation_signal,
    ]

    signals: list[RegimeSignal] = []
    for builder in signal_builders:
        try:
            sig = builder(client)
            if sig is not None:
                signals.append(sig)
        except Exception as exc:
            logger.warning("Signal builder %s failed: %s", builder.__name__, exc)

    if not signals:
        return RegimeAssessment(
            regime=Regime.NEUTRAL,
            confidence=0.0,
            signals=[],
            summary="No prediction-market data was available to assess the macro regime.",
        )

    # Weighted score in [-1, 1]
    total_weight = sum(s.weight for s in signals)
    raw_score = sum(s.weight * _DIRECTION_SCORE[s.signal_direction] for s in signals)
    score = raw_score / total_weight if total_weight else 0.0

    regime = _score_to_regime(score)
    confidence = min(abs(score) / 0.45, 1.0)  # 0.45 = strong threshold

    label = _REGIME_LABELS[regime]
    summary = (
        f"The overall macro environment appears {label} based on prediction-market consensus "
        f"(weighted score {score:+.2f}). "
        f"Assessment is based on {len(signals)} signal(s) with a combined weight of {total_weight:.2f}."
    )

    return RegimeAssessment(
        regime=regime,
        confidence=round(confidence, 2),
        signals=signals,
        summary=summary,
    )
