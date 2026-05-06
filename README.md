# Crypto Sentiment Model

This project runs a local Python pipeline that collects crypto-related RSS items, cleans and filters records, scores sentiment with Ollama, saves results to JSONL/CSV files and outputs a pdf report.

## What this project does

- Fetches RSS content from configured crypto news/community feeds
- Builds raw and cleaned datasets with reject reasons
- Deduplicates records using a stable hash
- Scores cleaned text with an Ollama model
- Exports scored output to JSONL and CSV
- Generates dashboard-like PDF report for summary statistics and visualization

## Prerequisites

- Python 3.8+
- Ollama installed on your machine
- A local Ollama model pulled (default: `llama3.2`)

## Setup

1. Install Ollama:
   - Download from [https://ollama.ai](https://ollama.ai), or
   - Use Homebrew: `brew install ollama`
2. Start Ollama:
   ```bash
   ollama serve
   ```
3. Pull model:
   ```bash
   ollama pull llama3.2
   ```
4. Install Python dependencies from project root:
   ```bash
   pip install -r requirements.txt
   ```

## Run the pipeline

From the `model` directory:

```bash
cd "model"
python3 run_ollama_pipeline.py
```

Optional model override:

```bash
cd "model"
PIPELINE_OLLAMA_MODEL=llama3.2 python3 run_ollama_pipeline.py
```

## Outputs

Each run generates date-based files in `model/data`:

- `raw/news_raw_YYYY-MM-DD.jsonl`
- `clean/news_clean_YYYY-MM-DD.jsonl`
- `clean/news_rejected_YYYY-MM-DD.jsonl`
- `scored/sentiment_YYYY-MM-DD.jsonl`
- `scored/sentiment_YYYY-MM-DD.csv`
- `reports/cleaning_stats_YYYY-MM-DD.json`
- `reports/report_YYYY-MM-DD.pdf`

## Project structure

- `README.md` - project overview and usage
- `requirements.txt` - Python dependencies
- `model/config.py` - keyword/feed configuration
- `model/rss_ingest.py` - RSS fetch + clean/filter logic
- `model/analyze.py` - Ollama scoring helpers used by pipeline
- `model/run_ollama_pipeline.py` - end-to-end local run script
- `evaluate/` - fixed-input model evaluation harness (see `evaluate/README.md`)

## Evaluate model performance (fixed input)

The `evaluate/` folder is designed to let you score a fixed clean dataset against a ground-truth label file and compare models/runs without generating lots of ad-hoc CSVs in the repo root.

- **Inputs** live in `evaluate/inputs/`:
  - `news_clean_2026-04-21.jsonl`
  - `sentiment_true_labels.csv`
- **Outputs** are written into per-run folders under `evaluate/output/`.

### Run scoring against the fixed input

From the project root:

```bash
python3 evaluate/run_eval.py --model llama3.2
```

This creates a new folder like:

- `evaluate/output/llama3.2/<run_id>/`
  - `scored_sentiment.jsonl`
  - `scored_sentiment.csv`
  - `labelled_sentiment.csv` (has `label` + `true_label` for evaluation)

It also updates:

- `evaluate/output/latest/` (so you can evaluate without having to copy a run path)

### Evaluate one run

```bash
python3 evaluate/evaluate_one.py
```

### Evaluate all runs + write summary

```bash
python3 evaluate/evaluate_all.py
```

This writes:

- `evaluate/output/<run>/metrics.json` for each run folder (next to `labelled_sentiment.csv`)

See `evaluate/README.md` for the full evaluation workflow.

## Configuration notes

- `config.py` controls RSS sources and keyword matching.
- `PIPELINE_OLLAMA_MODEL` can override the default model for a single run.

## Troubleshooting

- If Ollama is not reachable, ensure `ollama serve` is running.
- If model is missing, run `ollama pull llama3.2`.
- If output looks empty, check the reject file and `cleaning_stats` report for filtering reasons.
