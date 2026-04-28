import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"

# Reuse existing model-scoring logic from the model module.
sys.path.insert(0, str(MODEL_DIR))
from analyze import OLLAMA_MODEL, score_clean_rows  # noqa: E402


def load_clean_payload(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        payload = []
        with input_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    payload.append(json.loads(line))
        return payload

    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be an array of objects.")

    raise ValueError("Input file must be .json or .jsonl.")


def save_jsonl(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_model_slug(model_name: str) -> str:
    # Keep filenames cross-platform safe (e.g., deepseek-r1:8b -> deepseek-r1-8b).
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
    return slug or "model"


async def run_pipeline(input_file: str, true_labels_file: str, model_name: str | None) -> None:
    model_to_use = model_name or OLLAMA_MODEL
    model_slug = safe_model_slug(model_to_use)
    input_path = SCRIPT_DIR / input_file
    true_labels_path = SCRIPT_DIR / true_labels_file

    scored_jsonl_path = SCRIPT_DIR / f"scored_sentiment_{model_slug}.jsonl"
    scored_csv_path = SCRIPT_DIR / f"scored_sentiment_{model_slug}.csv"
    labelled_csv_path = SCRIPT_DIR / f"labelled_sentiment_{model_slug}.csv"

    clean_payload = load_clean_payload(input_path)
    print(f"Loaded clean payload: {len(clean_payload)} rows from {input_path.name}")

    print(f"Scoring with Ollama model: {model_to_use}...")
    scored_payload = await score_clean_rows(clean_payload, model_to_use)
    save_jsonl(scored_payload, scored_jsonl_path)
    print(f"Saved scored JSONL: {scored_jsonl_path.name}")

    scored_df = pd.DataFrame(scored_payload)
    if "text_for_model" in scored_df.columns:
        scored_df["text_for_model"] = (
            scored_df["text_for_model"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )
    scored_df.to_csv(scored_csv_path, index=False)
    print(f"Saved scored CSV: {scored_csv_path.name}")

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
    print(f"Saved labelled CSV: {labelled_csv_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ollama sentiment pipeline against clean news and emit labelled CSV in evaluate folder."
    )
    parser.add_argument(
        "--input",
        default="news_clean_2026-04-21.jsonl",
        help="Input clean news file (.json or .jsonl) inside evaluate/.",
    )
    parser.add_argument(
        "--true-labels",
        default="sentiment_true_labels.csv",
        help="Ground-truth label CSV inside evaluate/.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama model override, e.g. llama3.2",
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.input, args.true_labels, args.model))


if __name__ == "__main__":
    main()
