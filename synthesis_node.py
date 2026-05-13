import logging
import os
import json
from typing import Dict, Any, TypedDict, Optional, List, Tuple
import numpy as np
import datetime
from pathlib import Path
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import warnings
from functools import lru_cache
from dataclasses import dataclass, field
import concurrent.futures

from social_sentiment_loader import get_normalized_sentiment_score
from dcf import get_normalized_valuation_score
from market_data_loader import get_normalized_technical_score
from analyst_ratings_loader import get_normalized_fundamental_score

logger = logging.getLogger(__name__)

class Signal(TypedDict):
    score: float
    confidence: float

class HorizonResult(TypedDict):
    final_score: float
    relative_score: float
    rating: str
    source_confidence: float
    agreement: Optional[float]
    adjusted_weights: Dict[str, float]
    disagreement_map: Dict[str, Any]

class PeerDist(TypedDict):
    n: int
    mean: float
    std: float

class SynthesisResult(TypedDict):
    ticker: str
    peers: List[str]
    peer_comparison_status: str
    market_regime: str
    horizons: Dict[str, HorizonResult]
    signal_breakdown: Dict[str, Signal]
    peer_distributions: Dict[str, PeerDist]
    rationale: str

@dataclass
class SynthesisConfig:
    horizon_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "short": {
            "sentiment": 0.50,
            "technicals": 0.40,
            "valuation": 0.05,
            "fundamentals": 0.05
        },
        "medium": {
            "fundamentals": 0.40,
            "valuation": 0.30,
            "technicals": 0.15,
            "sentiment": 0.15
        },
        "long": {
            "fundamentals": 0.60,
            "valuation": 0.40,
            "technicals": 0.0,
            "sentiment": 0.0
        }
    })
    horizon_windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "short": (5, 20),
        "medium": (20, 90),
        "long": (90, 500)
    })
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Bull-Calm": {"technicals": +0.05, "sentiment": +0.05, "fundamentals": -0.05, "valuation": -0.05},
        "Bull-Volatile": {"sentiment": +0.10, "valuation": +0.05, "technicals": -0.05, "fundamentals": -0.10},
        "Bear-Calm": {"fundamentals": +0.10, "valuation": +0.10, "technicals": -0.10, "sentiment": -0.10},
        "Bear-Volatile": {"sentiment": +0.10, "fundamentals": +0.05, "valuation": +0.05, "technicals": -0.20},
        "Unknown": {"fundamentals": 0.0, "valuation": 0.0, "technicals": 0.0, "sentiment": 0.0}
    })
    rating_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Strong Buy": 0.8,
        "Buy": 0.6,
        "Hold": 0.4,
        "Sell": 0.2
    })
    conflict_floor: float = 0.15
    conflict_multiplier: float = 1.5
    max_peers: int = 5

@lru_cache(maxsize=1)
def _detect_market_regime_cached(date_key: str) -> str:
    try:
        ticker = yf.Ticker("^GSPC")
        end_date = datetime.date.fromisoformat(date_key)
        start_date = end_date - datetime.timedelta(days=730)
        hist = ticker.history(start=start_date, end=end_date)
        
        hist = hist.dropna()
        if hist.empty or len(hist) < 200:
            logger.warning("Insufficient market data for HMM.")
            return "Unknown"
        
        hist['log_return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['volatility'] = hist['log_return'].rolling(window=20).std()
        
        hist = hist.dropna()
        
        X = hist[['log_return', 'volatility']].values
        
        best_hmm = None
        best_bic = float('inf')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n in [2, 3, 4]:
                hmm = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
                try:
                    hmm.fit(X)
                    logL = hmm.score(X)
                    n_features = 2
                    n_params = n**2 - 1 + 2 * n * n_features
                    bic = -2 * logL + n_params * np.log(X.shape[0])
                    if bic < best_bic:
                        best_bic = bic
                        best_hmm = hmm
                except Exception as e:
                    logger.debug(f"HMM fit failed for n={n}: {e}")
                    continue

        if best_hmm is None:
            return "Unknown"

        hidden_states = best_hmm.predict(X)
        recent_states = hidden_states[-5:]
        current_state = int(np.bincount(recent_states).argmax())
        
        means = best_hmm.means_
        
        avg_mean = np.median(means[:, 0])
        avg_vol = np.median(means[:, 1])
        
        curr_mean = means[current_state, 0]
        curr_vol = means[current_state, 1]
        
        direction = "Bull" if curr_mean >= avg_mean else "Bear"
        volatility = "Volatile" if curr_vol >= avg_vol else "Calm"
        
        return f"{direction}-{volatility}"
    except Exception as e:
        logger.error(f"Failed to detect market regime: {e}")
        return "Unknown"


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


class SynthesisNode:
    def __init__(self, config: SynthesisConfig = None):
        self.config = config or SynthesisConfig()
        self._fetch_cache = {}

    def _detect_market_regime(self) -> str:
        date_key = str(datetime.datetime.now(datetime.UTC).date())
        return _detect_market_regime_cached(date_key)
        
    def _safe_fetch(self, name: str, fn, ticker: str) -> Signal:
        date_key = str(datetime.datetime.now(datetime.UTC).date())
        cache_key = (name, ticker, date_key)
        
        if cache_key in self._fetch_cache:
            return self._fetch_cache[cache_key]

        try:
            sig = fn(ticker)
            if not isinstance(sig, dict) or "score" not in sig or "confidence" not in sig:
                raise ValueError(f"Invalid signal shape: {sig!r}")
            if not (0 <= sig["score"] <= 1 and 0 <= sig["confidence"] <= 1):
                raise ValueError(f"Signal out of [0,1]: {sig!r}")
            self._fetch_cache[cache_key] = sig
            return sig
        except Exception as e:
            logger.warning(f"{name} loader failed for {ticker}: {e}")
            res = {"score": 0.5, "confidence": 0.0}
            self._fetch_cache[cache_key] = res
            return res

    def _get_peers(self, ticker: str) -> List[str]:
        competitors = []
        try:
            peer_path = Path(__file__).parent / "sp500_competitors.json"
            if peer_path.exists():
                with open(peer_path, "r") as f:
                    data = json.load(f)
                    if ticker in data:
                        competitors = data[ticker].get("competitors", [])[:self.config.max_peers]
        except Exception as e:
            logger.error(f"Failed to load competitors: {e}")
        return competitors

    def _calculate_horizon(self, 
                           signals: Dict[str, Signal], 
                           base_weights: Dict[str, float], 
                           market_regime: str,
                           peer_dists: Dict[str, PeerDist]) -> HorizonResult:
        
        adjusted_weights = base_weights.copy()
        adjustments = self.config.regime_adjustments.get(market_regime, self.config.regime_adjustments.get("Unknown"))
        
        for k, v in adjustments.items():
            if k in adjusted_weights and adjusted_weights[k] > 0:
                adjusted_weights[k] = max(0.0, adjusted_weights[k] + v)

        total_adj = sum(adjusted_weights.values())
        if total_adj > 0:
            adjusted_weights = {k: v/total_adj for k, v in adjusted_weights.items()}

        sum_weighted_scores = 0.0
        sum_relative_scores = 0.0
        total_effective_weight = 0.0
        total_source_confidence = 0.0
        
        scores_list = []
        eff_weights_list = []
        
        for key, weight in adjusted_weights.items():
            if weight == 0.0:
                continue
                
            sig = signals[key]
            raw_score = sig["score"]
            conf = sig["confidence"]
            
            # Z-score logic for cross-sectional rank
            if key in peer_dists and peer_dists[key]["std"] > 0 and peer_dists[key]["n"] >= 2:
                z_score = (raw_score - peer_dists[key]["mean"]) / peer_dists[key]["std"]
                relative_score = sigmoid(z_score)
            else:
                relative_score = raw_score
                
            effective_weight = weight * conf
            sum_weighted_scores += raw_score * effective_weight
            sum_relative_scores += relative_score * effective_weight
            total_effective_weight += effective_weight
            
            total_source_confidence += conf * weight
            scores_list.append(relative_score)
            eff_weights_list.append(effective_weight)

        if total_effective_weight == 0:
            final_raw = 0.5
            final_rel = 0.5
        else:
            final_raw = sum_weighted_scores / total_effective_weight
            final_rel = sum_relative_scores / total_effective_weight
            
        scores_array = np.array(scores_list)
        eff_weights_array = np.array(eff_weights_list)
        
        active_signals = np.sum(eff_weights_array > 0.001)
        sum_eff = np.sum(eff_weights_array)
        if sum_eff > 0 and active_signals >= 2:
            norm_w = eff_weights_array / sum_eff
            mean_w = np.sum(scores_array * norm_w)
            dispersion = float(np.sqrt(np.sum(norm_w * (scores_array - mean_w)**2)))
            
            # Catch bimodal wide gaps that pure std misses
            spread = float(np.max(scores_array) - np.min(scores_array))
            if spread > 0.3:
                dispersion = max(dispersion, spread * 0.5)
                
            agreement = round(1.0 - min(dispersion, 1.0), 3)
            median_score = float(np.median(scores_array))
        else:
            dispersion = 0.0
            agreement = None
            median_score = 0.5 if len(scores_list) == 0 else float(np.median(scores_array))
            
        threshold = max(dispersion * self.config.conflict_multiplier, self.config.conflict_floor)
        
        high_conflict = []
        if sum_eff > 0:
            for k, s in signals.items():
                if k in adjusted_weights and adjusted_weights[k] > 0:
                    score = s["score"]
                    if k in peer_dists and peer_dists[k]["std"] > 0 and peer_dists[k]["n"] >= 2:
                        z_score = (score - peer_dists[k]["mean"]) / peer_dists[k]["std"]
                        score = sigmoid(z_score)
                    if abs(score - median_score) > threshold:
                        high_conflict.append(k)

        rating = self._get_rating_label(final_rel)

        return {
            "final_score": round(final_raw, 3),
            "relative_score": round(final_rel, 3),
            "rating": rating,
            "source_confidence": round(float(total_source_confidence), 3),
            "agreement": agreement,
            "adjusted_weights": {k: round(v, 3) for k, v in adjusted_weights.items()},
            "disagreement_map": {
                "dispersion": round(dispersion, 3),
                "dynamic_threshold": round(threshold, 3),
                "anchor": round(median_score, 3),
                "high_conflict_sources": high_conflict,
                "n_effective_signals": int(active_signals)
            }
        }

    def calculate_synthesis(self, ticker: str) -> SynthesisResult:
        ticker = ticker.upper()
        logger.info(f"Starting Multi-Horizon Signal Synthesis for {ticker}...")

        market_regime = self._detect_market_regime()
        logger.info(f"Detected Market Regime: {market_regime}")

        peers = self._get_peers(ticker)
        all_tickers = [ticker] + peers
        logger.info(f"Pulling parallel cross-section for group: {all_tickers}")

        tasks = {
            "fundamentals": get_normalized_fundamental_score,
            "valuation": get_normalized_valuation_score,
            "technicals": get_normalized_technical_score,
            "sentiment": get_normalized_sentiment_score
        }
        
        # Cross-sectional network loop: Fetch all signals for Ticker + Peers
        pool_results = {t: {} for t in all_tickers}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(all_tickers) * 4)) as executor:
            future_map = {
                executor.submit(self._safe_fetch, source, fn, t): (t, source)
                for t in all_tickers
                for source, fn in tasks.items()
            }
            for future in concurrent.futures.as_completed(future_map):
                t, source = future_map[future]
                pool_results[t][source] = future.result()

        target_signals = pool_results[ticker]
        
        # Calculate cross-sectional distributions (EXCLUDING target ticker)
        peer_distributions: Dict[str, PeerDist] = {}
        for source in tasks.keys():
            valid_scores = [pool_results[t][source]["score"] 
                            for t in peers 
                            if pool_results[t][source]["confidence"] > 0]
            n = len(valid_scores)
            if n > 1:
                peer_distributions[source] = {
                    "n": n,
                    "mean": float(np.mean(valid_scores)),
                    "std": max(float(np.std(valid_scores, ddof=1)), 0.1)
                }
            else:
                peer_distributions[source] = {"n": n, "mean": 0.5, "std": 0.0}
                
        # Determine peer comparison status
        if len(peers) == 0:
            comp_status = "no_peers"
        else:
            has_enough = any(dist["n"] >= 2 for dist in peer_distributions.values())
            comp_status = "ok" if has_enough else "insufficient_data"

        # Calculate horizons
        horizons = {}
        for h_name, h_weights in self.config.horizon_weights.items():
            horizons[h_name] = self._calculate_horizon(
                target_signals, h_weights, market_regime, peer_distributions
            )

        rationale = self._generate_rationale(ticker, market_regime, horizons, comp_status)

        payload: SynthesisResult = {
            "ticker": ticker,
            "peers": peers,
            "peer_comparison_status": comp_status,
            "market_regime": market_regime,
            "horizons": horizons,
            "signal_breakdown": target_signals,
            "peer_distributions": {k: {"n": v["n"], "mean": round(v["mean"],3), "std": round(v["std"],3)} for k, v in peer_distributions.items()},
            "rationale": rationale
        }
        
        return payload

    def _get_rating_label(self, score: float) -> str:
        for label, threshold in sorted(self.config.rating_thresholds.items(), key=lambda kv: kv[1], reverse=True):
            if score >= threshold: return label
        return "Strong Sell"

    def _generate_rationale(self, ticker: str, regime: str, horizons: Dict[str, HorizonResult], comp_status: str) -> str:
        short_r = horizons["short"]["rating"]
        med_r = horizons["medium"]["rating"]
        long_r = horizons["long"]["rating"]
        
        summary = f"The synthesis node evaluated {ticker} across multiple horizons in a {regime} market regime.\n"
        
        long_raw = horizons["long"]["final_score"]
        long_rel = horizons["long"]["relative_score"]

        if comp_status == "ok":
            summary += f"Cross-Sectional Rank: Objectively ranked against direct industry peers.\n"
            if abs(long_raw - long_rel) > 0.2:
                quartile = "top quartile" if long_rel >= 0.75 else "bottom quartile" if long_rel <= 0.25 else "middle"
                summary += f"Valuation Note: {ticker}'s absolute long-term score is {long_raw:.3f}, but it ranks in the {quartile} of its peer set, driving the {long_r} relative rating.\n"
        elif comp_status == "no_peers":
            summary += "Note: No direct competitors defined. Scores fallback to absolute values instead of peer-relative percentiles.\n"
        else:
            summary += "Note: Insufficient valid data from competitors. Scores fallback to absolute values.\n"

        def rating_to_int(r):
            sorted_labels = sorted(self.config.rating_thresholds.items(), key=lambda kv: kv[1])
            mapping = {label: i+1 for i, (label, _) in enumerate(sorted_labels)}
            mapping["Strong Sell"] = 0
            return mapping.get(r, len(mapping)//2)
            
        s_int, l_int = rating_to_int(short_r), rating_to_int(long_r)
        
        if s_int > l_int:
            summary += f"Trajectory: WARNING (Short/long divergence). {ticker} is a {short_r} in the short-term ({self.config.horizon_windows['short'][0]}-{self.config.horizon_windows['short'][1]} trading days) driven by technical/sentiment momentum, but deteriorates to a {long_r} long-term ({self.config.horizon_windows['long'][0]}-{self.config.horizon_windows['long'][1]} days) heavily weighted by core fundamentals and DCF valuation.\n"
        elif s_int < l_int:
            summary += f"Trajectory: IMPROVING. {ticker} is constrained to a {short_r} in the short-term bounded by weak momentum, but strengthens to a {long_r} rating long-term based on resilient structural upside.\n"
        else:
            summary += f"Trajectory: STABLE. {ticker} maintains a consistent {short_r} rating across effectively all time horizons.\n"
            
        med = horizons["medium"]
        if med["agreement"] is None:
            summary += "Note: Valid cross-sectional signals were absent or lacked confidence."
        elif med["agreement"] < 0.5:
            summary += "There is high conflict across data sources in the medium-term outlook."
            
        return summary

if __name__ == "__main__":
    import argparse
    import json
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    
    node = SynthesisNode()
    result = node.calculate_synthesis(args.ticker)
    print(json.dumps(result, indent=2))
