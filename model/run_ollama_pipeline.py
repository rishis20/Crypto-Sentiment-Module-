"""
Run the local CSV-only RSS -> Ollama sentiment pipeline.

Outputs:
- data/raw/news_raw_YYYY-MM-DD.jsonl
- data/clean/news_clean_YYYY-MM-DD.jsonl
- data/scored/sentiment_YYYY-MM-DD.jsonl
- data/scored/sentiment_YYYY-MM-DD.csv
"""

import asyncio
import os
from datetime import datetime, timezone
import pandas as pd

from analyze import OLLAMA_MODEL, analyze_text_sentiment
from rss_ingest import collect_raw_rss_records, collect_rss_articles, fetch_feed_items, save_jsonl

PROMPT_VERSION = "v1.0.0"


def ensure_dirs() -> dict:
    base = "data"
    raw_dir = os.path.join(base, "raw")
    clean_dir = os.path.join(base, "clean")
    scored_dir = os.path.join(base, "scored")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(scored_dir, exist_ok=True)
    return {"raw": raw_dir, "clean": clean_dir, "scored": scored_dir}


def score_to_label(score: float) -> str:
    if score <= -0.2:
        return "bearish"
    if score >= 0.2:
        return "bullish"
    return "neutral"


def build_explanation(label: str, score: float) -> str:
    if label == "bullish":
        return f"Tone indicates positive market momentum (score {score:.3f})."
    if label == "bearish":
        return f"Tone indicates negative market pressure (score {score:.3f})."
    return f"Signals are mixed or informational (score {score:.3f})."


async def run_pipeline(model_name: str | None = None) -> None:
    model_to_use = model_name or OLLAMA_MODEL
    dirs = ensure_dirs()
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_path = os.path.join(dirs["raw"], f"news_raw_{date_tag}.jsonl")
    clean_path = os.path.join(dirs["clean"], f"news_clean_{date_tag}.jsonl")
    scored_jsonl_path = os.path.join(dirs["scored"], f"sentiment_{date_tag}.jsonl")
    scored_csv_path = os.path.join(dirs["scored"], f"sentiment_{date_tag}.csv")

    print("Step 1/4: Fetching RSS feeds (single pass)...")
    feed_items = fetch_feed_items()
    print(f"  Fetched entries: {len(feed_items)}")

    print("Step 2/4: Building raw snapshot records...")
    raw_records = collect_raw_rss_records(feed_items)
    raw_payload = [
        {
            "source": r.source,
            "url": r.url,
            "published_at": r.published_at,
            "fetched_at": r.fetched_at,
            "title_raw": r.title_raw,
            "summary_raw": r.summary_raw,
            "raw_payload": r.raw_payload,
        }
        for r in raw_records
    ]
    save_jsonl(raw_payload, raw_path)
    print(f"  Saved raw records: {len(raw_payload)} -> {raw_path}")

    print("Step 3/4: Building clean records...")
    clean_records = collect_rss_articles(feed_items)
    clean_payload = [
        {
            "source": r.source.lower(),
            "url": r.url,
            "published_at": r.published_date,
            "fetched_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "title_clean": r.title,
            "summary_clean": r.summary,
            "crypto": r.crypto,
            "text_for_model": r.text_for_sentiment,
        }
        for r in clean_records
    ]
    save_jsonl(clean_payload, clean_path)
    print(f"  Saved clean records: {len(clean_payload)} -> {clean_path}")

    if not clean_records:
        print("No RSS records matched configured crypto keywords. Skipping scoring.")
        return

    print("Step 4/4: Scoring with Ollama...")
    scored_payload = []
    for item in clean_payload:
        score = await analyze_text_sentiment(item["text_for_model"], model_to_use)
        # Retry once if score is malformed/out of contract.
        if not isinstance(score, (int, float)) or score < -1.0 or score > 1.0:
            score = await analyze_text_sentiment(item["text_for_model"], model_to_use)
        # Final contract enforcement.
        if not isinstance(score, (int, float)):
            score = 0.0
        score = max(-1.0, min(1.0, float(score)))
        label = score_to_label(score)
        confidence = round(min(1.0, max(0.0, abs(score))), 3)
        scored_payload.append(
            {
                **item,
                "score": round(float(score), 6),
                "label": label,
                "confidence": confidence,
                "explanation": build_explanation(label, float(score)),
                "model_name": model_to_use,
                "prompt_version": PROMPT_VERSION,
                "scored_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }
        )
    save_jsonl(scored_payload, scored_jsonl_path)
    print(f"  Saved scored JSONL: {len(scored_payload)} -> {scored_jsonl_path}")

    print("Finalizing: Exporting CSV for comparison...")
    pd.DataFrame(scored_payload).to_csv(scored_csv_path, index=False)
    print(f"  Saved scored CSV: {scored_csv_path}")


if __name__ == "__main__":
    # Optional model override from env var if provided.
    model_override = os.getenv("PIPELINE_OLLAMA_MODEL")
    asyncio.run(run_pipeline(model_override))
