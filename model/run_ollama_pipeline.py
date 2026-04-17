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
import json
from datetime import datetime, timezone
import pandas as pd

from analyze import OLLAMA_MODEL, score_clean_rows
from rss_ingest import (
    build_clean_dataset,
    collect_raw_rss_records,
    fetch_feed_items,
    save_jsonl,
    MIN_ALPHA_RATIO,
    MIN_CHAR_COUNT,
)


def ensure_dirs() -> dict:
    base = "data"
    raw_dir = os.path.join(base, "raw")
    clean_dir = os.path.join(base, "clean")
    scored_dir = os.path.join(base, "scored")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(scored_dir, exist_ok=True)
    return {"raw": raw_dir, "clean": clean_dir, "scored": scored_dir}


async def run_pipeline(model_name: str | None = None) -> None:
    model_to_use = model_name or OLLAMA_MODEL
    dirs = ensure_dirs()
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_path = os.path.join(dirs["raw"], f"news_raw_{date_tag}.jsonl")
    clean_path = os.path.join(dirs["clean"], f"news_clean_{date_tag}.jsonl")
    reject_path = os.path.join(dirs["clean"], f"news_rejected_{date_tag}.jsonl")
    scored_jsonl_path = os.path.join(dirs["scored"], f"sentiment_{date_tag}.jsonl")
    scored_csv_path = os.path.join(dirs["scored"], f"sentiment_{date_tag}.csv")
    reports_dir = os.path.join("data", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stats_path = os.path.join(reports_dir, f"cleaning_stats_{date_tag}.json")
    fetched_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    print("Step 1/4: Fetching RSS feeds (single pass)...")
    feed_items = fetch_feed_items()
    print(f"  Fetched entries: {len(feed_items)}")

    print("Step 2/4: Building raw snapshot records...")
    raw_records = collect_raw_rss_records(feed_items)
    raw_payload = [
        {
            "source": r.source,
            "feed_url": r.feed_url,
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
    clean_payload, rejected_payload, clean_stats = build_clean_dataset(
        prefetched_items=feed_items,
        clean_dir=dirs["clean"],
        fetched_at=fetched_at,
    )
    save_jsonl(clean_payload, clean_path)
    save_jsonl(rejected_payload, reject_path)
    print(f"  Saved clean records: {len(clean_payload)} -> {clean_path}")
    print(f"  Saved rejected records: {len(rejected_payload)} -> {reject_path}")

    if not clean_payload:
        print("No RSS records matched configured crypto keywords. Skipping scoring.")
        return

    print("Step 4/4: Scoring with Ollama...")
    scored_payload = await score_clean_rows(clean_payload, model_to_use)
    save_jsonl(scored_payload, scored_jsonl_path)
    print(f"  Saved scored JSONL: {len(scored_payload)} -> {scored_jsonl_path}")

    print("Finalizing: Exporting CSV for comparison...")
    pd.DataFrame(scored_payload).to_csv(scored_csv_path, index=False)
    print(f"  Saved scored CSV: {scored_csv_path}")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": date_tag,
                "fetched_at": fetched_at,
                "totals": {
                    "raw_rows": len(raw_payload),
                    "clean_rows": len(clean_payload),
                    "rejected_rows": len(rejected_payload),
                    "scored_rows": len(scored_payload),
                },
                "by_source": {
                    "raw": clean_stats.get("by_source_raw", {}),
                    "clean": clean_stats.get("by_source_clean", {}),
                },
                "reject_reason_counts": clean_stats.get("reject_reason_counts", {}),
                "min_char_count": MIN_CHAR_COUNT,
                "min_alpha_ratio": MIN_ALPHA_RATIO,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Saved cleaning stats JSON: {stats_path}")


if __name__ == "__main__":
    # Optional model override from env var if provided.
    model_override = os.getenv("PIPELINE_OLLAMA_MODEL")
    asyncio.run(run_pipeline(model_override))
