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
from rss_ingest import save_jsonl


def ensure_dirs() -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir, "data")
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
    scored_jsonl_path = os.path.join(dirs["scored"], f"sentiment_{date_tag}_{model_to_use}.jsonl")
    scored_csv_path = os.path.join(dirs["scored"], f"sentiment_{date_tag}_{model_to_use}.csv")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "news_clean_2026-04-21.jsonl")
    clean_payload = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            clean_payload.append(json.loads(line))

    print("Scoring with Ollama...")
    scored_payload = await score_clean_rows(clean_payload, model_to_use)
    save_jsonl(scored_payload, scored_jsonl_path)
    print(f"  Saved scored JSONL: {len(scored_payload)} -> {scored_jsonl_path}")

    print("Finalizing: Exporting CSV for comparison...")
    scored_df = pd.DataFrame(scored_payload)
    if "text_for_model" in scored_df.columns:
        scored_df["text_for_model"] = (
            scored_df["text_for_model"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )
    scored_df.to_csv(scored_csv_path, index=False)
    print(f"  Saved scored CSV: {scored_csv_path}")

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(eval_dir)
    true_labels_path = os.path.join(eval_dir, "sentiment_true_labels.csv")
    labelled_csv_path = os.path.join(project_root, f"labelled_sentiment_{model_to_use}.csv")

    true_labels_df = pd.read_csv(true_labels_path)
    labelled_df = pd.concat(
        [
            scored_df[["text_for_model", "label"]]
                .rename(columns={"text_for_model": "text"})
                .reset_index(drop=True),
            true_labels_df[["true_label"]].reset_index(drop=True),
        ],
        axis=1,
    )
    labelled_df.to_csv(labelled_csv_path, index=False)
    print(f"  Saved labelled CSV: {labelled_csv_path}")

if __name__ == "__main__":
    # Optional model override from env var if provided.
    model_override = os.getenv("PIPELINE_OLLAMA_MODEL")
    asyncio.run(run_pipeline(model_override))
