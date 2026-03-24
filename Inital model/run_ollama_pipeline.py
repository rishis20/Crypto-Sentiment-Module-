"""
Run the local CSV-only RSS -> Ollama sentiment pipeline.

Outputs:
- rss_articles.csv
- rss_articles_scoring_input.csv
- rss_articles_scored.csv
"""

import asyncio
import os
import pandas as pd

from analyze import process_csv_file
from rss_ingest import collect_rss_articles, save_to_csv

RAW_RSS_CSV = "rss_articles.csv"
SCORING_INPUT_CSV = "rss_articles_scoring_input.csv"
SCORED_CSV = "rss_articles_scored.csv"


def build_scoring_input(raw_csv: str, scoring_csv: str) -> int:
    df = pd.read_csv(raw_csv)
    if df.empty:
        raise ValueError("No RSS records found to score.")
    if "text_for_sentiment" not in df.columns:
        raise ValueError("Missing 'text_for_sentiment' column in RSS CSV.")

    # analyze.py scores the first column. Keep text first, preserve metadata columns.
    ordered_columns = ["text_for_sentiment"] + [c for c in df.columns if c != "text_for_sentiment"]
    scoring_df = df[ordered_columns]
    scoring_df.to_csv(scoring_csv, index=False)
    return len(scoring_df)


async def run_pipeline(model_name: str | None = None) -> None:
    print("Step 1/3: Collecting RSS articles...")
    records = collect_rss_articles()
    save_to_csv(records, RAW_RSS_CSV)
    print(f"  Saved raw RSS records: {len(records)} -> {RAW_RSS_CSV}")

    if not records:
        print("No RSS records matched configured crypto keywords. Skipping scoring.")
        return

    print("Step 2/3: Preparing scoring input CSV...")
    rows = build_scoring_input(RAW_RSS_CSV, SCORING_INPUT_CSV)
    print(f"  Prepared {rows} rows -> {SCORING_INPUT_CSV}")

    print("Step 3/3: Scoring with Ollama via analyze.py...")
    output_path = await process_csv_file(SCORING_INPUT_CSV, SCORED_CSV, model_name)
    print(f"  Scored CSV saved: {output_path}")


if __name__ == "__main__":
    # Optional model override from env var if provided.
    model_override = os.getenv("PIPELINE_OLLAMA_MODEL")
    asyncio.run(run_pipeline(model_override))
