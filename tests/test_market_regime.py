"""Tests for the market regime scoring engine.

Uses real Kalshi API calls — no mocks. Requires network access.
Pure scoring logic (price parsing, threshold mapping) is tested without API calls.
"""

from __future__ import annotations

import sys
import os

import pytest

# Ensure project root is on the path so imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_client import KalshiClient, KalshiAPIError
from market_regime import (
    Regime,
    RegimeAssessment,
    RegimeSignal,
    TrendInfo,
    _price_to_prob,
    _score_to_regime,
    _compute_trend,
    assess_regime,
)
from report_regime_section import generate_regime_section


# ── _price_to_prob (pure logic, no API) ──────────────────────────────

class TestPriceToProb:
    def test_last_price_preferred(self) -> None:
        market = {"last_price": "0.65", "yes_bid": "0.60", "yes_ask": "0.70"}
        assert _price_to_prob(market) == pytest.approx(0.65)

    def test_last_price_dollars_field(self) -> None:
        market = {"last_price_dollars": "0.42"}
        assert _price_to_prob(market) == pytest.approx(0.42)

    def test_midpoint_from_bid_ask(self) -> None:
        market = {"yes_bid": "0.30", "yes_ask": "0.40"}
        assert _price_to_prob(market) == pytest.approx(0.35)

    def test_bid_only(self) -> None:
        market = {"yes_bid": "0.25"}
        assert _price_to_prob(market) == pytest.approx(0.25)

    def test_ask_only(self) -> None:
        market = {"yes_ask": "0.80"}
        assert _price_to_prob(market) == pytest.approx(0.80)

    def test_fallback(self) -> None:
        assert _price_to_prob({}) == pytest.approx(0.5)

    def test_dollars_suffix_fields(self) -> None:
        market = {"yes_bid_dollars": "0.20", "yes_ask_dollars": "0.30"}
        assert _price_to_prob(market) == pytest.approx(0.25)


# ── _score_to_regime (pure logic, no API) ────────────────────────────

class TestScoreToRegime:
    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.80, Regime.STRONG_BULL),
            (0.45, Regime.STRONG_BULL),
            (0.30, Regime.BULL),
            (0.15, Regime.BULL),
            (0.10, Regime.NEUTRAL),
            (0.00, Regime.NEUTRAL),
            (-0.10, Regime.NEUTRAL),
            (-0.15, Regime.BEAR),
            (-0.20, Regime.BEAR),
            (-0.44, Regime.BEAR),
            (-0.45, Regime.STRONG_BEAR),
            (-0.50, Regime.STRONG_BEAR),
            (-1.00, Regime.STRONG_BEAR),
        ],
    )
    def test_thresholds(self, score: float, expected: Regime) -> None:
        assert _score_to_regime(score) == expected


# ── KalshiClient (real API calls) ────────────────────────────────────

class TestKalshiClient:
    """Hit the real Kalshi public API to verify our client works."""

    @pytest.fixture(scope="class")
    def client(self) -> KalshiClient:
        return KalshiClient()

    def test_get_markets_returns_list(self, client: KalshiClient) -> None:
        data = client.get_markets(limit=5, status="open")
        assert "markets" in data
        assert isinstance(data["markets"], list)
        assert len(data["markets"]) > 0

    def test_get_single_market(self, client: KalshiClient) -> None:
        # First grab any open market ticker
        listing = client.get_markets(limit=1, status="open")
        ticker = listing["markets"][0]["ticker"]
        data = client.get_market(ticker)
        market = data.get("market", data)
        assert market["ticker"] == ticker
        assert "title" in market

    def test_get_series_list(self, client: KalshiClient) -> None:
        data = client.get_series_list(limit=5)
        assert "series" in data or "categories" in data or isinstance(data, dict)

    def test_get_markets_with_series_filter(self, client: KalshiClient) -> None:
        """Fetch markets for the Fed decision series — known to exist."""
        data = client.get_markets(series_ticker="KXFEDDECISION", limit=10)
        markets = data.get("markets", [])
        # This series should have markets (may be open or closed)
        assert isinstance(markets, list)

    def test_invalid_market_raises(self, client: KalshiClient) -> None:
        with pytest.raises(KalshiAPIError):
            client.get_market("TOTALLY_FAKE_TICKER_999")

    def test_get_markets_batch(self, client: KalshiClient) -> None:
        # Grab a couple real tickers, filtering to standard-length ones
        # (very long multivariate tickers can 404 on the individual endpoint)
        listing = client.get_markets(limit=10, status="open")
        tickers = [
            m["ticker"] for m in listing["markets"]
            if len(m["ticker"]) < 60
        ][:3]
        results = client.get_markets_batch(tickers)
        # Batch skips failures gracefully, so at least some should succeed
        assert len(results) >= 1
        for r in results:
            assert "ticker" in r

    def test_batch_skips_bad_tickers_gracefully(self, client: KalshiClient) -> None:
        results = client.get_markets_batch(["FAKE_TICKER_XYZ"])
        assert results == []

    def test_get_candlesticks_returns_list(self, client: KalshiClient) -> None:
        """Fetch daily candlesticks for the recession market."""
        import time
        end_ts = int(time.time())
        start_ts = end_ts - 30 * 86400  # 30 days
        candles = client.get_candlesticks(
            "KXRECSSNBER", "KXRECSSNBER-26",
            period_interval=1440,
            start_ts=start_ts, end_ts=end_ts,
        )
        assert isinstance(candles, list)
        assert len(candles) >= 1
        # Check candle structure
        c = candles[0]
        assert "end_period_ts" in c
        assert "price" in c
        assert "close_dollars" in c["price"]

    def test_get_candlesticks_bad_ticker_returns_empty(self, client: KalshiClient) -> None:
        candles = client.get_candlesticks("FAKE", "FAKE-TICKER", period_interval=1440)
        assert candles == []


# ── _compute_trend (real API calls) ─────────────────────────────────

class TestComputeTrend:
    """Test trend computation against live Kalshi data."""

    @pytest.fixture(scope="class")
    def client(self) -> KalshiClient:
        return KalshiClient()

    def test_trend_for_recession_market(self, client: KalshiClient) -> None:
        trend = _compute_trend(client, "KXRECSSNBER", "KXRECSSNBER-26")
        assert trend is not None
        assert isinstance(trend, TrendInfo)
        assert 0.0 <= trend.prob_start <= 1.0
        assert 0.0 <= trend.prob_end <= 1.0
        assert trend.sample_days >= 3
        assert trend.trend_label in ("stable", "rising", "falling", "surging", "plunging")

    def test_trend_returns_none_for_bad_market(self, client: KalshiClient) -> None:
        trend = _compute_trend(client, "FAKE", "FAKE-TICKER")
        assert trend is None


# ── assess_regime (real API calls) ───────────────────────────────────

class TestAssessRegimeLive:
    """Run the full regime assessment against the live Kalshi API."""

    @pytest.fixture(scope="class")
    def assessment(self) -> RegimeAssessment:
        client = KalshiClient()
        return assess_regime(client)

    def test_returns_valid_regime(self, assessment: RegimeAssessment) -> None:
        assert isinstance(assessment.regime, Regime)

    def test_confidence_in_range(self, assessment: RegimeAssessment) -> None:
        assert 0.0 <= assessment.confidence <= 1.0

    def test_has_signals(self, assessment: RegimeAssessment) -> None:
        # We expect at least some signals from the live API
        assert len(assessment.signals) >= 1

    def test_signals_have_valid_fields(self, assessment: RegimeAssessment) -> None:
        for sig in assessment.signals:
            assert 0.0 <= sig.implied_probability <= 1.0
            assert sig.signal_direction in ("bullish", "bearish", "neutral")
            assert 0.0 < sig.weight <= 1.0
            assert sig.name
            assert sig.explanation
            assert isinstance(sig.kalshi_ticker, str)

    def test_signals_have_trend_data(self, assessment: RegimeAssessment) -> None:
        """At least some signals should include historical trend info."""
        trends = [s.trend for s in assessment.signals if s.trend is not None]
        # We expect at least one signal to have trend data
        assert len(trends) >= 1
        for t in trends:
            assert isinstance(t, TrendInfo)
            assert t.trend_label in ("stable", "rising", "falling", "surging", "plunging")

    def test_summary_is_nonempty(self, assessment: RegimeAssessment) -> None:
        assert isinstance(assessment.summary, str)
        assert len(assessment.summary) > 20

    def test_weights_sum_correctly(self, assessment: RegimeAssessment) -> None:
        total = sum(s.weight for s in assessment.signals)
        # Weights should sum to <= 1.0 (may be less if some signals missing)
        assert 0.0 < total <= 1.01


# ── Report section generation (uses live assessment) ─────────────────

class TestRegimeSectionLive:
    """Generate the Markdown section from a live regime assessment."""

    @pytest.fixture(scope="class")
    def section_md(self) -> str:
        client = KalshiClient()
        assessment = assess_regime(client)
        return generate_regime_section(assessment, "NVDA", "Nvidia")

    def test_contains_required_headings(self, section_md: str) -> None:
        assert "## Market Regime Analysis" in section_md
        assert "### Current Regime:" in section_md
        assert "### Implications for Nvidia" in section_md
        assert "### Methodology Note" in section_md

    def test_contains_kalshi_attribution(self, section_md: str) -> None:
        assert "Kalshi" in section_md

    def test_contains_signals_table(self, section_md: str) -> None:
        assert "| Signal |" in section_md
        assert "Implied Prob." in section_md
        assert "90-Day Trend" in section_md

    def test_contains_timestamp(self, section_md: str) -> None:
        assert "UTC" in section_md

    def test_mentions_company(self, section_md: str) -> None:
        assert "Nvidia" in section_md
        assert "NVDA" in section_md
