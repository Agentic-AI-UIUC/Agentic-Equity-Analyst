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
import chromadb
from chromadb.utils import embedding_functions

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

        self.chroma_collection_name = "social_sentiment"
        self.persist_dir = "./chroma"
        self._init_chroma()

    def _init_chroma(self):
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("Missing OPENAI_API_KEY in environment for Chroma embeddings. Ensure it is set.")
            return
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.chroma_collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

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
        """Scrape Reddit asynchronously to get post data and top comments."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        extracted_data = []

        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15.0) as client:
            for sub in self.subreddits:
                search_url = f"https://old.reddit.com/r/{sub}/search.json?q={ticker}&restrict_sr=on&sort=new&limit={limit_posts}"
                logging.info(f"Searching {sub} for {ticker}: {search_url}")
                
                try:
                    resp = await self._fetch_with_backoff(client, search_url)
                    if resp.status_code == 429:
                        logging.warning("Rate limit hit (429) during search! Stopping early to avoid IP ban.")
                        break
                    if resp.status_code != 200:
                        logging.error(f"Failed to fetch {search_url}, status: {resp.status_code}")
                        continue
                    
                    data = resp.json()
                    children = data.get("data", {}).get("children", [])
                    
                    for child in children:
                        post_data = child.get("data", {})
                        permalink = post_data.get("permalink")
                        if not permalink:
                            continue
                        
                        post_id = post_data.get("id")
                        title = post_data.get("title", "")
                        body = post_data.get("selftext", "")
                        url = f"https://www.reddit.com{permalink}"
                        
                        post_content = {
                            "id": post_id,
                            "title": title,
                            "body": body,
                            "url": url,
                            "source": f"r/{sub}",
                            "timestamp": post_data.get("created_utc"),
                            "top_comments": []
                        }
                        
                        comments_url = f"https://old.reddit.com{permalink}.json?sort=top&limit={limit_comments}"
                        await asyncio.sleep(2)  # Base throttling for Reddit parsing safety
                        
                        c_resp = await self._fetch_with_backoff(client, comments_url)
                        if c_resp.status_code == 429:
                            logging.warning("Rate limit hit (429) during comment fetch! Saving what we have and aborting.")
                            extracted_data.append(post_content)
                            return extracted_data
                        
                        if c_resp.status_code == 200:
                            c_data = c_resp.json()
                            if isinstance(c_data, list) and len(c_data) > 1:
                                comments_tree = c_data[1].get("data", {}).get("children", [])
                                for c_tree in comments_tree[:limit_comments]:
                                    cData = c_tree.get("data", {})
                                    c_body = cData.get("body")
                                    if c_body and c_body not in ["[deleted]", "[removed]"]:
                                        post_content["top_comments"].append(c_body)
                                        
                        extracted_data.append(post_content)

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

    def process_and_store(self, ticker: str, extracted_data: List[Dict[str, Any]]) -> int:
        """Analyze sentiments and upsert to chroma DB consistently."""
        if not hasattr(self, 'collection'):
            logging.error("Collection unavailable. Aborting process.")
            return 0
            
        if not extracted_data:
            logging.info("No data extracted to process.")
            return 0
        
        documents = []
        metadatas = []
        ids = []

        ts_insert_ny = now_ny()
        date_retrieved = ts_insert_ny.date().isoformat()
        time_retrieved = ts_insert_ny.strftime("%H:%M:%S")

        for item in extracted_data:
            combined_text = f"{item['title']}\n\n{item['body']}"
            if item['top_comments']:
                comments_text = "\n---\n".join(item['top_comments'])
                combined_text += f"\n\nTop Comments:\n{comments_text}"
            
            sentiment_result = self.analyze_sentiment(combined_text)

            ts_pub_ny = ensure_ny_timestamp(item['timestamp'])
            date_published = ts_pub_ny.date().isoformat()
            time_published = ts_pub_ny.strftime("%H:%M:%S")

            doctype = "social_media"

            composite_key = (
                f"{ticker.upper()}_"
                f"{item['source']}_"
                f"{date_published}_"
                f"{item['id']}" # Reddit post id prevents replication 
            )

            meta = {
                "symbol": ticker.upper(),
                "date_retrieved": date_retrieved,
                "time_retrieved": time_retrieved,
                "date_published": date_published,
                "time_published": time_published,
                "source": item['source'],
                "url": item['url'],
                "title": item['title'],
                "doctype": doctype,
                "metadata_key": composite_key,
                "finbert_label": sentiment_result["label"],
                "finbert_score": float(sentiment_result["score"]),
                "finbert_composite": float(sentiment_result["composite_score"])
            }


            documents.append(combined_text)
            metadatas.append(meta)
            ids.append(composite_key)

        try:
            # Replaced uuid4 append to ensure stable composite ID upsertions 
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            return len(ids)
        except Exception as e:
            logging.error(f"Failed to upsert to ChromaDB: {e}")
            raise e

    async def run(self, ticker: str, limit_posts: int = 5, limit_comments: int = 3):
        logging.info(f"Starting Social Sentiment Loader for {ticker}...")
        data = await self.fetch_reddit_data(ticker, limit_posts, limit_comments)
        count = self.process_and_store(ticker, data)
        logging.info(f"Successfully scraped and upserted {count} items for {ticker} into 'social_sentiment' collection.")


# --- Exposing Analytical Functionality ---

_CHROMA_CLIENT = None
_EF = None

def get_social_sentiment_collection():
    global _CHROMA_CLIENT, _EF
    import os
    import chromadb
    from chromadb.utils import embedding_functions

    if not _CHROMA_CLIENT:
        _CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma")
    if not _EF:
        _EF = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
    return _CHROMA_CLIENT.get_or_create_collection(
        name="social_sentiment",
        embedding_function=_EF,
        metadata={"hnsw:space": "cosine"}
    )

def analyze_social_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Public API callable to query stored sentiment and return aggregated metrics.
    Retrieves the most recent documents for a ticker from ChromaDB and groups signals.
    """
    try:
        from collections import Counter
        import re
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return {"error": "Chroma database unavailable (is OPENAI_API_KEY set?)."}
            
        collection = get_social_sentiment_collection()
        
        # Fetch the exact matches from the DB using the metadata filter (last 7 days to keep score fresh)
        cutoff_date = (now_ny() - dt.timedelta(days=7)).date().isoformat()
        
        results = collection.get(
            where={"$and": [
                {"symbol": ticker.upper()},
                {"date_published": {"$gte": cutoff_date}}
            ]}
        )
        
        if not results['documents']:
            return {"ticker": ticker.upper(), "message": f"No social sentiment data found for the last 7 days."}
            
        docs = results['documents']
        metadatas = results['metadatas']
        
        total_composite = sum(meta.get('finbert_composite', 0.0) for meta in metadatas)
        avg_score = total_composite / len(metadatas) if metadatas else 0.0

        # High level label
        sentiment_label = "Neutral"
        if avg_score > 0.2:
            sentiment_label = "Positive"
        elif avg_score < -0.2:
            sentiment_label = "Negative"

        # Trending keywords extraction (excluding generic terms and formatting characters)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "top", "comments"}
        words = []
        for d in docs:
            cleaned = re.sub(r'[^a-zA-Z\s]', '', d.lower())
            tokens = [w for w in cleaned.split() if w not in stop_words and len(w) > 3]
            words.extend(tokens)
            
        common_words = [word for word, count in Counter(words).most_common(10)]

        return {
            "ticker": ticker.upper(),
            "volume_tracked": len(docs),
            "average_composite_score": round(avg_score, 3),
            "sentiment_label": sentiment_label,
            "trending_topics": common_words,
            "recent_titles": [m.get('title') for m in metadatas[:3]]
        }

    except Exception as e:
        return {"error": f"Failed compiling sentiment: {e}"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--limit-posts", type=int, default=5, help="Number of posts per subreddit")
    parser.add_argument("--limit-comments", type=int, default=3, help="Number of top comments to retrieve per post")
    args = parser.parse_args()

    loader = SocialSentimentLoader()
    asyncio.run(loader.run(ticker=args.ticker, limit_posts=args.limit_posts, limit_comments=args.limit_comments))
    
    # Showcase metrics
    metrics = analyze_social_sentiment(args.ticker)
    import json
    print(f"\nSentiment Metrics for {args.ticker}:")
    print(json.dumps(metrics, indent=2))
