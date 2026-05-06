import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"

# Reuse existing model-scoring logic, files are in the 'model' directory
sys.path.insert(0, str(MODEL_DIR))
from analyze import OLLAMA_MODEL, score_clean_rows


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


def default_run_id() -> str:
    # Human-readable (easy to copy/paste and recognize).
    # Example: run_2026-05-06__21-03-12
    return datetime.now().strftime("run_%Y-%m-%d__%H-%M-%S")

def write_latest(output_dir: Path, run_dir: Path, model_name: str, run_id: str) -> None:
    """
    Maintain a stable, easy-to-find pointer for "most recent run".

    This avoids forcing users to understand/remember run folder naming.
    We copy (not symlink) for cross-platform friendliness.
    """
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    for name in ["scored_sentiment.jsonl", "scored_sentiment.csv", "labelled_sentiment.csv"]:
        src = run_dir / name
        if src.exists():
            (latest_dir / name).write_bytes(src.read_bytes())

    (latest_dir / "run_info.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model_name,
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


async def run_pipeline(
    input_file: str,
    true_labels_file: str,
    model_name: str | None,
    output_dir: str,
    run_id: str | None,
) -> None:
    model_to_use = model_name or OLLAMA_MODEL
    model_slug = safe_model_slug(model_to_use)

    inputs_dir = SCRIPT_DIR / "inputs"
    input_path = (inputs_dir / input_file) if not Path(input_file).is_absolute() else Path(input_file)
    true_labels_path = (
        (inputs_dir / true_labels_file)
        if not Path(true_labels_file).is_absolute()
        else Path(true_labels_file)
    )

    base_output_dir = SCRIPT_DIR / output_dir
    resolved_run_id = run_id or default_run_id()
    # Clear structure: evaluate/output/<model_slug>/<run_id>/
    run_dir = base_output_dir / model_slug / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scored_jsonl_path = run_dir / "scored_sentiment.jsonl"
    scored_csv_path = run_dir / "scored_sentiment.csv"
    labelled_csv_path = run_dir / "labelled_sentiment.csv"

    clean_payload = load_clean_payload(input_path)
    print(f"Loaded clean payload: {len(clean_payload)} rows from {input_path}")

    print(f"Scoring with Ollama model: {model_to_use}...")
    scored_payload = await score_clean_rows(clean_payload, model_to_use)
    save_jsonl(scored_payload, scored_jsonl_path)
    print(f"Saved scored JSONL: {scored_jsonl_path}")

    scored_df = pd.DataFrame(scored_payload)
    if "text_for_model" in scored_df.columns:
        scored_df["text_for_model"] = (
            scored_df["text_for_model"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )
    scored_df.to_csv(scored_csv_path, index=False)
    print(f"Saved scored CSV: {scored_csv_path}")

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
    print(f"Saved labelled CSV: {labelled_csv_path}")

    write_latest(base_output_dir, run_dir, model_to_use, resolved_run_id)
    print(f"Updated latest run pointer: {base_output_dir / 'latest'}")
    print(f"Run id: {resolved_run_id}")
    print(f"Run folder: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the fixed clean input and write outputs into evaluate/output/<run>/."
    )
    parser.add_argument(
        "--input",
        default="news_clean_2026-04-21.jsonl",
        help="Input clean news file (.json or .jsonl) inside evaluate/inputs/ (or absolute path).",
    )
    parser.add_argument(
        "--true-labels",
        default="sentiment_true_labels.csv",
        help="Ground-truth label CSV inside evaluate/inputs/ (or absolute path).",
    )
    parser.add_argument("--model", default=None, help="Optional Ollama model override, e.g. llama3.2")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory (relative to evaluate/) to write run folders into.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id override (default is timestamp).")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.input, args.true_labels, args.model, args.output_dir, args.run_id))


if __name__ == "__main__":
    main()

