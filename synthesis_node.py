import logging
from typing import Dict, Any, TypedDict, Optional
import numpy as np
import datetime
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

class SynthesisResult(TypedDict):
    ticker: str
    final_score: float
    rating: str
    source_confidence: float
    agreement: Optional[float]
    market_regime: str
    signal_breakdown: Dict[str, Signal]
    adjusted_weights: Dict[str, float]
    disagreement_map: Dict[str, Any]
    rationale: str

@dataclass
class SynthesisConfig:
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "fundamentals": 0.40,
        "valuation": 0.30,
        "technicals": 0.15,
        "sentiment": 0.15
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

class SynthesisNode:
    def __init__(self, config: SynthesisConfig = None):
        self.config = config or SynthesisConfig()

    def _detect_market_regime(self) -> str:
        date_key = str(datetime.datetime.now(datetime.timezone.utc).date())
        return _detect_market_regime_cached(date_key)
        
    def _safe_fetch(self, name: str, fn, ticker: str) -> Signal:
        try:
            sig = fn(ticker)
            if not isinstance(sig, dict) or "score" not in sig or "confidence" not in sig:
                raise ValueError(f"Invalid signal shape: {sig!r}")
            if not (0 <= sig["score"] <= 1 and 0 <= sig["confidence"] <= 1):
                raise ValueError(f"Signal out of [0,1]: {sig!r}")
            return sig
        except Exception as e:
            logger.warning(f"{name} loader failed for {ticker}: {e}")
            return {"score": 0.5, "confidence": 0.0}

    def calculate_synthesis(self, ticker: str) -> SynthesisResult:
        ticker = ticker.upper()
        logger.info(f"Starting Weighted Signal Synthesis for {ticker}...")

        market_regime = self._detect_market_regime()
        logger.info(f"Detected Market Regime: {market_regime}")

        adjusted_weights = self.config.base_weights.copy()
        adjustments = self.config.regime_adjustments.get(market_regime, self.config.regime_adjustments.get("Unknown"))
        for k, v in adjustments.items():
            if k in adjusted_weights:
                adjusted_weights[k] += v

        total_adj = sum(adjusted_weights.values())
        if total_adj > 0:
            adjusted_weights = {k: v/total_adj for k, v in adjusted_weights.items()}

        tasks = {
            "fundamentals": get_normalized_fundamental_score,
            "valuation": get_normalized_valuation_score,
            "technicals": get_normalized_technical_score,
            "sentiment": get_normalized_sentiment_score
        }
        signals = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_source = {
                executor.submit(self._safe_fetch, source, fn, ticker): source 
                for source, fn in tasks.items()
            }
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                signals[source] = future.result()

        sum_weighted_scores = 0.0
        total_effective_weight = 0.0
        total_source_confidence = 0.0
        scores_list = []
        eff_weights_list = []
        
        for key, weight in adjusted_weights.items():
            sig = signals[key]
            score = sig["score"]
            conf = sig["confidence"]
            
            effective_weight = weight * conf
            sum_weighted_scores += score * effective_weight
            total_effective_weight += effective_weight
            
            total_source_confidence += conf * weight
            scores_list.append(score)
            eff_weights_list.append(effective_weight)

        if total_effective_weight == 0:
            weighted_score = 0.5
        else:
            weighted_score = sum_weighted_scores / total_effective_weight
            
        scores_array = np.array(scores_list)
        eff_weights_array = np.array(eff_weights_list)
        
        sum_eff = np.sum(eff_weights_array)
        if sum_eff > 0:
            norm_w = eff_weights_array / sum_eff
            mean_w = np.sum(scores_array * norm_w)
            dispersion = float(np.sqrt(np.sum(norm_w * (scores_array - mean_w)**2)))
            agreement = round(1.0 - min(dispersion, 1.0), 3)
        else:
            dispersion = 0.0
            agreement = None
            
        median_score = float(np.median(scores_array))
        threshold = max(dispersion * self.config.conflict_multiplier, self.config.conflict_floor)
        
        high_conflict = [
            k for k, s in signals.items() 
            if abs(s["score"] - median_score) > threshold
        ]

        source_confidence = float(total_source_confidence)

        rating = self._get_rating_label(weighted_score)

        return {
            "ticker": ticker,
            "final_score": round(weighted_score, 3),
            "rating": rating,
            "source_confidence": round(source_confidence, 3),
            "agreement": agreement,
            "market_regime": market_regime,
            "signal_breakdown": signals,
            "adjusted_weights": {k: round(v, 3) for k, v in adjusted_weights.items()},
            "disagreement_map": {
                "dispersion": round(dispersion, 3),
                "dynamic_threshold": round(threshold, 3),
                "anchor": round(median_score, 3),
                "high_conflict_sources": high_conflict
            },
            "rationale": self._generate_rationale(ticker, rating, weighted_score, source_confidence, agreement, high_conflict, market_regime, adjusted_weights, signals)
        }

    def _get_rating_label(self, score: float) -> str:
        for label, threshold in self.config.rating_thresholds.items():
            if score >= threshold: return label
        return "Strong Sell"

    def _generate_rationale(self, ticker, rating, score, source_confidence, agreement, high_conflict, regime, adjusted_weights, signals) -> str:
        summary = f"The synthesis node issued a {rating} rating for {ticker} with a conviction score of {score:.2f} in a {regime} market regime."
        
        if source_confidence < 0.4:
            summary += " Note: Underlying data quality/confidence is LOW."
        elif source_confidence > 0.7:
            summary += " Source confidence is HIGH."
            
        if agreement is None:
            summary += " The signals are absent or entirely lacking confidence."
        elif agreement < 0.5:
            summary += " However, there is significant disagreement between components."
        elif agreement >= 0.8:
            summary += " The signals are broadly aligned."
            
        if high_conflict:
            sources = ", ".join(high_conflict)
            summary += f" Key conflict(s) detected in: {sources}."
            
        def contribution(k):
            eff_w = adjusted_weights[k] * signals[k]["confidence"]
            return eff_w * abs(signals[k]["score"] - 0.5)
            
        primary = max(adjusted_weights, key=contribution)
        summary += f" Under this configuration, {primary} was the strongest directional driver (highest conviction-weighted contribution)."
        
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
