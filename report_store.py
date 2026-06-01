"""ChromaDB-backed report history store with local file fallback."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Max chars to embed (text-embedding-3-small max ~8191 tokens ≈ 32k chars; use 6k to be safe)
_MAX_EMBED_CHARS = 6000


def _get_collection():
    """Return the chromadb report_history collection, or None if unavailable."""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        store = Chroma(
            database=os.getenv("CHROMADB"),
            collection_name="report_history",
            embedding_function=embeddings,
            chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
            tenant=os.getenv("CHROMADB_TENANT"),
        )
        # Pre-compute an embedding so we can add full-text docs with our own embedding
        return store, embeddings
    except Exception:
        return None, None


def save_report(report_text: str, ticker: str, company: str, year: str) -> str:
    """
    Persist a report to ChromaDB (primary) and local disk (always).
    Returns the report ID.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_company = company.replace(" ", "_")
    report_id = f"{ticker.upper()}_{safe_company}_{year}_{ts}"

    # Always save locally as fallback / for offline use
    local_path = REPORTS_DIR / f"{report_id}.txt"
    local_path.write_text(report_text, encoding="utf-8")
    Path("report.txt").write_text(report_text, encoding="utf-8")

    # Attempt ChromaDB save
    store, embeddings = _get_collection()
    if store is not None:
        try:
            # Embed only the first N chars; store the full text as the document
            embed_vector = embeddings.embed_query(report_text[:_MAX_EMBED_CHARS])
            store._collection.add(
                ids=[report_id],
                embeddings=[embed_vector],
                documents=[report_text],
                metadatas=[{
                    "ticker": ticker.upper(),
                    "company": company,
                    "year": year,
                    "timestamp": ts,
                }],
            )
        except Exception as exc:
            # ChromaDB save failed — local file is the fallback
            print(f"[report_store] ChromaDB save failed (using local file): {exc}")

    return report_id


def list_reports() -> list[dict]:
    """
    Return all saved reports as a list of dicts with keys:
      id, ticker, company, year, timestamp
    Prefers ChromaDB; falls back to local files.
    """
    store, _ = _get_collection()
    if store is not None:
        try:
            results = store._collection.get(include=["metadatas"])
            rows = []
            for rid, meta in zip(results["ids"], results["metadatas"]):
                rows.append({"id": rid, **meta})
            return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)
        except Exception as exc:
            print(f"[report_store] ChromaDB list failed (falling back to local): {exc}")

    # Local fallback
    rows = []
    for path in sorted(REPORTS_DIR.glob("*.txt"), reverse=True):
        parts = path.stem.split("_")
        if len(parts) >= 4:
            ticker = parts[0]
            ts = f"{parts[-2]}_{parts[-1]}"
            year = parts[-3]
            company = " ".join(parts[1:-3])
        else:
            ticker = company = year = ts = ""
        rows.append({
            "id": path.stem,
            "ticker": ticker,
            "company": company,
            "year": year,
            "timestamp": ts.replace("_", ""),
        })
    return rows


def get_report(report_id: str) -> Optional[str]:
    """
    Retrieve full report text by ID.
    Prefers ChromaDB; falls back to local file.
    """
    store, _ = _get_collection()
    if store is not None:
        try:
            results = store._collection.get(ids=[report_id], include=["documents"])
            if results["documents"]:
                return results["documents"][0]
        except Exception as exc:
            print(f"[report_store] ChromaDB get failed (falling back to local): {exc}")

    # Local fallback
    local_path = REPORTS_DIR / f"{report_id}.txt"
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    return None


def format_report_label(r: dict) -> str:
    """Human-readable label for a report metadata dict."""
    ticker = r.get("ticker", "")
    company = r.get("company", "")
    year = r.get("year", "")
    ts = r.get("timestamp", "")
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        date_str = dt.strftime("%b %d, %Y  %I:%M %p")
    except ValueError:
        date_str = ts
    return f"{ticker} — {company} {year}  |  {date_str}"
