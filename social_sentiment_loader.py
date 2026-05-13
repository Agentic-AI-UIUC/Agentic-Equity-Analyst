"""
social_sentiment_loader.py
--------------------------
Scrapes Reddit for mentions of specific tickers to compile Crowd Intelligence.
Applies FinBERT for financial sentiment analysis and persists to ChromaDB.
"""

from __future__ import annotations
import os
import datetime as dt
import asyncio
import logging
from typing import List, Dict, Any, Optional

import httpx
import pytz
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer
import re
from collections import Counter
import math
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NY_TZ = pytz.timezone("America/New_York")

def now_ny() -> dt.datetime:
    return dt.datetime.now(NY_TZ)

def ensure_ny_timestamp(ts: Optional[float]) -> dt.datetime:
    """Takes a UTC timestamp (like those from Reddit) and converts to NY timezone dt."""
    if not ts:
        return now_ny()
    utc_dt = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    return utc_dt.astimezone(NY_TZ)

def sanitize_for_key(s: str) -> str:
    """Make a URL/file-ish string safe for the composite key."""
    return s.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_")


class SocialSentimentLoader:
    def __init__(self, subreddits: List[str] = None):
        self.subreddits = subreddits or ["wallstreetbets", "stocks", "investing"]
        load_dotenv()
        
        logging.info("Loading FinBERT model and Tokenizer. This might take a few seconds...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer=self.tokenizer)
        logging.info("FinBERT loaded successfully.")

    async def _fetch_with_backoff(self, client: httpx.AsyncClient, url: str, max_retries: int = 3) -> httpx.Response:
        """Fetch a URL with exponential backoff on transient errors (5xx/timeouts/etc)."""
        retries = 0
        backoff = 2
        while retries <= max_retries:
            try:
                resp = await client.get(url)
                if resp.status_code == 429:
                    return resp  # Handle 429 safely in the caller
                
                if resp.status_code >= 500:
                    logging.warning(f"Server error {resp.status_code} for {url}. Retrying in {backoff}s...")
                else:
                    return resp
            except httpx.RequestError as e:
                logging.warning(f"Request exception: {e} for {url}. Retrying in {backoff}s...")
            
            await asyncio.sleep(backoff)
            retries += 1
            backoff *= 2
        
        raise Exception(f"Failed to fetch {url} after {max_retries} retries.")

    async def fetch_reddit_data(self, ticker: str, limit_posts: int = 5, limit_comments: int = 3) -> List[Dict[str, Any]]:
        """Scrape Reddit asynchronously with concurrent comment fetching."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        extracted_data = []
        semaphore = asyncio.Semaphore(3) # Limit concurrency to avoid IP blocks

        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15.0) as client:
            for sub in self.subreddits:
                search_url = f"https://old.reddit.com/r/{sub}/search.json?q={ticker}&restrict_sr=on&sort=new&limit={limit_posts}"
                logging.info(f"Searching {sub} for {ticker}: {search_url}")
                
                try:
                    resp = await self._fetch_with_backoff(client, search_url)
                    if resp.status_code == 429:
                        logging.warning(f"Rate limit hit during search for {sub}!")
                        continue
                    if resp.status_code != 200:
                        continue
                    
                    data = resp.json()
                    children = data.get("data", {}).get("children", [])

                    async def fetch_comments_for_post(child):
                        async with semaphore:
                            post_data = child.get("data", {})
                            permalink = post_data.get("permalink")
                            if not permalink: return None
                            
                            post_content = {
                                "id": post_data.get("id"),
                                "title": post_data.get("title", ""),
                                "body": post_data.get("selftext", ""),
                                "url": f"https://www.reddit.com{permalink}",
                                "source": f"r/{sub}",
                                "timestamp": post_data.get("created_utc"),
                                "top_comments": []
                            }
                            
                            comments_url = f"https://old.reddit.com{permalink}.json?sort=top&limit={limit_comments}"
                            await asyncio.sleep(0.75) # Respectful throttle
                            
                            try:
                                c_resp = await self._fetch_with_backoff(client, comments_url)
                                if c_resp.status_code == 200:
                                    c_data = c_resp.json()
                                    if isinstance(c_data, list) and len(c_data) > 1:
                                        comments_tree = c_data[1].get("data", {}).get("children", [])
                                        for c_node in comments_tree[:limit_comments]:
                                            c_body = c_node.get("data", {}).get("body")
                                            if c_body and c_body not in ["[deleted]", "[removed]"]:
                                                post_content["top_comments"].append(c_body)
                            except Exception as e:
                                logging.error(f"Error fetching comments for {permalink}: {e}")
                            
                            return post_content

                    # Concurrent fetch for all posts in this subreddit
                    tasks = [fetch_comments_for_post(c) for c in children]
                    results = await asyncio.gather(*tasks)
                    extracted_data.extend([r for r in results if r])

                except Exception as e:
                    logging.error(f"Exception during extraction for {sub}: {e}")
                    
        return extracted_data

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Runs FinBERT on unified text cleanly truncating at the tokenizer limit."""
        if not text.strip():
            return {"label": "neutral", "score": 0.0, "composite_score": 0.0}
        
        try:
            # Tokenize correctly limiting strictly to max token length (512 for FinBERT)
            encoded = self.tokenizer.encode(text, truncation=True, max_length=512)
            decoded_text = self.tokenizer.decode(encoded, skip_special_tokens=True)
            
            res = self.sentiment_pipeline(decoded_text)[0]
            label = res['label']
            score = res['score']
            
            if label.lower() == 'positive':
                composite = score
            elif label.lower() == 'negative':
                composite = -score
            else:
                composite = 0.0
                
            return {
                "label": label,
                "score": score,
                "composite_score": composite
            }
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            raise e

    async def run(self, ticker: str, limit_posts: int = 5, limit_comments: int = 3) -> Dict[str, Any]:
        """Runs the full pipeline: fetch, analyze, and aggregate."""
        logging.info(f"Starting Social Sentiment Loader for {ticker}...")
        extracted_data = await self.fetch_reddit_data(ticker, limit_posts, limit_comments)
        
        if not extracted_data:
            return {"ticker": ticker.upper(), "message": "No social sentiment data found."}
            
        results = []
        for item in extracted_data:
            combined_text = f"{item['title']}\n\n{item['body']}"
            if item['top_comments']:
                comments_text = "\n---\n".join(item['top_comments'])
                combined_text += f"\n\nTop Comments:\n{comments_text}"
            
            sentiment_result = self.analyze_sentiment(combined_text)
            results.append({
                "text": combined_text,
                "title": item['title'],
                "sentiment": sentiment_result
            })
            
        return self.aggregate_results(ticker, results)

    def aggregate_results(self, ticker: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates individual sentiment results into a summary report."""
        if not results:
            return {"ticker": ticker.upper(), "message": "No data to aggregate."}

        total_composite = sum(r['sentiment']['composite_score'] for r in results)
        avg_score = total_composite / len(results)

        sentiment_label = "Neutral"
        if avg_score > 0.15:
            sentiment_label = "Positive"
        elif avg_score < -0.15:
            sentiment_label = "Negative"

        # Trending keywords extraction
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "top", "comments", "reddit", ticker.lower()}
        words = []
        for r in results:
            cleaned = re.sub(r'[^a-zA-Z\s]', '', r['text'].lower())
            tokens = [w for w in cleaned.split() if w not in stop_words and len(w) > 3]
            words.extend(tokens)
            
        common_words = [word for word, count in Counter(words).most_common(10)]

        # Calculate variance/std_dev of sentiment scores
        composite_scores = [r['sentiment']['composite_score'] for r in results]
        std_dev = float(np.std(composite_scores)) if len(composite_scores) > 1 else 0.0

        return {
            "ticker": ticker.upper(),
            "volume_tracked": len(results),
            "average_composite_score": round(avg_score, 3),
            "sentiment_std_dev": round(std_dev, 3),
            "sentiment_label": sentiment_label,
            "trending_topics": common_words,
            "recent_titles": [r['title'] for r in results[:3]],
            "timestamp": now_ny().isoformat()
        }


# --- Exposing Analytical Functionality ---

_LOADER_INSTANCE: Optional[SocialSentimentLoader] = None

def get_sentiment_loader() -> SocialSentimentLoader:
    """Singleton accessor for the SocialSentimentLoader to reuse FinBERT."""
    global _LOADER_INSTANCE
    if _LOADER_INSTANCE is None:
        _LOADER_INSTANCE = SocialSentimentLoader()
    return _LOADER_INSTANCE

_SESSION_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRATION_HOURS = 6

def get_on_demand_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Unified entry point: Checks in-memory session cache, then scrapes if needed.
    """
    ticker = ticker.upper()
    now = now_ny()

    # 1. Session Cache Check
    if ticker in _SESSION_CACHE:
        cached_data = _SESSION_CACHE[ticker]
        cached_time = dt.datetime.fromisoformat(cached_data["timestamp"])
        if (now - cached_time).total_seconds() < CACHE_EXPIRATION_HOURS * 3600:
            logging.info(f"Using session-cached social sentiment for {ticker}.")
            return cached_data

    # 2. Scrape Live
    logging.info(f"No fresh cache for {ticker}. Running on-demand scrape...")
    loader = get_sentiment_loader()
    try:
        # Run async scraper in a sync-friendly way
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        
        metrics = asyncio.run(loader.run(ticker, limit_posts=5, limit_comments=3))
        
        # 3. Store in session cache
        if "message" not in metrics:
            _SESSION_CACHE[ticker] = metrics
            
        return metrics
    except Exception as e:
        logging.error(f"On-demand scrape failed: {e}")
        return {"ticker": ticker, "error": str(e)}

def get_normalized_sentiment_score(ticker: str) -> Dict[str, Any]:
    """
    Returns a normalized sentiment score (0.0-1.0) derived from Reddit data.
    0.0 = Strong Negative, 0.5 = Neutral, 1.0 = Strong Positive.
    """
    metrics = get_on_demand_sentiment(ticker)
    if "error" in metrics or "average_composite_score" not in metrics:
        return {"score": 0.5, "confidence": 0.0, "details": metrics}
    
    raw_score = metrics["average_composite_score"]
    # Map [-1, 1] to [0, 1] using Sigmoid-like transition or simple linear for sentiment
    # For sentiment, keeping it (raw+1)/2 is often fine, but let's use a soft sigmoid as requested
    normalized = 1 / (1 + math.exp(-raw_score * 5)) # Scale by 5 to make it sensitive
    
    # NEW: Agreement metric based on std_dev
    std_dev = metrics.get("sentiment_std_dev", 0.5)
    agreement = 1 - min(std_dev, 1.0)
    
    # NEW: Non-linear confidence curve
    volume = metrics.get("volume_tracked", 0)
    confidence = (1 - math.exp(-volume / 5)) * agreement
    
    return {
        "score": round(normalized, 3),
        "confidence": round(confidence, 3),
        "raw_value": raw_score,
        "std_dev": std_dev,
        "label": metrics.get("sentiment_label", "Neutral")
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--on-demand", action="store_true", help="Run the full on-demand flow")
    args = parser.parse_args()

    if args.on_demand:
        metrics = get_on_demand_sentiment(args.ticker)
    else:
        loader = get_sentiment_loader()
        metrics = asyncio.run(loader.run(ticker=args.ticker))
    
    import json
    print(f"\nSentiment Metrics for {args.ticker}:")
    print(json.dumps(metrics, indent=2))
