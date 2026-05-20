"""Minimal Kalshi public API client. No authentication needed."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Daily candle interval (minutes).
_PERIOD_DAILY = 1440

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Small delay (seconds) between consecutive batch calls to respect rate limits.
_BATCH_DELAY = 0.25


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an unexpected HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Kalshi API {status_code}: {detail}")


class KalshiClient:
    """Fetches public market data from Kalshi's API.

    Uses a ``requests.Session`` with automatic retries on transient failures
    (429, 500, 502, 503, 504).  No authentication headers are required for
    the public endpoints used here.
    """

    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a GET request and return the parsed JSON response."""
        url = f"{self._base_url}{path}"
        logger.debug("GET %s params=%s", url, params)
        resp = self._session.get(url, params=params, timeout=self._timeout)
        if not resp.ok:
            raise KalshiAPIError(resp.status_code, resp.text[:300])
        return resp.json()

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_series_list(self, **params: Any) -> dict[str, Any]:
        """GET /series - list available series.

        Optional params: ``limit``, ``cursor``, ``category``, etc.
        """
        return self._get("/series", params=params or None)

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        """GET /series/{series_ticker} - single series info."""
        return self._get(f"/series/{series_ticker}")

    def get_markets(self, **params: Any) -> dict[str, Any]:
        """GET /markets - list markets with optional filters.

        Useful params: ``series_ticker``, ``status``, ``limit``, ``cursor``.
        """
        return self._get("/markets", params=params or None)

    def get_market(self, ticker: str) -> dict[str, Any]:
        """GET /markets/{ticker} - single market detail."""
        return self._get(f"/markets/{ticker}")

    # ------------------------------------------------------------------
    # Candlestick / history helpers
    # ------------------------------------------------------------------

    def get_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        *,
        period_interval: int = _PERIOD_DAILY,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch daily candlestick data for a market.

        Parameters
        ----------
        series_ticker : str
            Parent series (e.g. ``"KXRECSSNBER"``).
        market_ticker : str
            Specific market (e.g. ``"KXRECSSNBER-26"``).
        period_interval : int
            Candle width in minutes.  1 = 1 min, 60 = hourly, 1440 = daily.
        start_ts, end_ts : int, optional
            Unix epoch bounds.  Defaults to last 90 days → now.

        Returns
        -------
        list[dict]
            Candlestick dicts with ``end_period_ts``, ``price``, ``volume_fp``, etc.
        """
        import time as _time

        if end_ts is None:
            end_ts = int(_time.time())
        if start_ts is None:
            start_ts = end_ts - 90 * 86400  # 90 days back

        params: dict[str, Any] = {
            "period_interval": period_interval,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        try:
            data = self._get(path, params=params)
            return data.get("candlesticks", [])
        except (KalshiAPIError, requests.RequestException) as exc:
            logger.warning("Could not fetch candlesticks for %s: %s", market_ticker, exc)
            return []

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def get_markets_batch(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple individual markets, respecting rate limits.

        Returns a list of market dicts.  Markets that fail to fetch are
        logged and skipped rather than raising.
        """
        results: list[dict[str, Any]] = []
        for i, ticker in enumerate(tickers):
            try:
                data = self.get_market(ticker)
                results.append(data.get("market", data))
            except KalshiAPIError as exc:
                logger.warning("Skipping market %s: %s", ticker, exc)
            except requests.RequestException as exc:
                logger.warning("Network error fetching %s: %s", ticker, exc)
            if i < len(tickers) - 1:
                time.sleep(_BATCH_DELAY)
        return results
