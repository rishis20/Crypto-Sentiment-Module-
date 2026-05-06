# Evaluation harness (`evaluate/`)

This folder is a **fixed-input evaluation harness** for comparing model performance. It uses a fixed input stored in the inputs folder and a manually labelled test set of 186 articles and reddit threads

## Folder layout

- `inputs/`
  - `news_clean_2026-04-21.jsonl`: fixed cleaned input dataset
  - `sentiment_true_labels.csv`: test set
- `output/`
  - `latest/` (always points to the most recent run)
  - `<model_slug>/<run_id>/`
    - `scored_sentiment.jsonl`: scored rows (JSONL)
    - `scored_sentiment.csv`: scored rows (CSV)
    - `labelled_sentiment.csv`: **the evaluation file** (has `label` and `true_label`)
    - `metrics.json`: metrics for that run (written by `evaluate_all.py`)
  - `summary.csv`: optional cross-run summary (only written with `--write-summary`)

Note: `output/` starts empty. It is generated when you run `run_eval.py` / `evaluate_all.py`.

## Before you run evaluation

- Start Ollama:

```bash
ollama serve
```

- Ensure your model is available (example):

```bash
ollama pull llama3.2
```

## Step 1: Run a scoring run (creates one run folder)

From project root:

```bash
python3 evaluate/run_eval.py --model llama3.2
```

What this does:

- Reads `evaluate/inputs/news_clean_2026-04-21.jsonl`
- Scores it with the model (`label` column)
- Writes outputs into `evaluate/output/<model_slug>/<run_id>/`
- Updates `evaluate/output/latest/` so you can evaluate without copying paths

Default `run_id` format is human-readable, e.g. `run_2026-05-06__21-03-12`.

## Step 2A: Evaluate a single run (prints report)

```bash
python3 evaluate/evaluate_one.py
```

If you want to evaluate a specific run folder:

```bash
python3 evaluate/evaluate_one.py evaluate/output/<model_slug>/<run_id>/labelled_sentiment.csv
```

## Step 2B: Evaluate all runs (writes per-run metrics next to the CSV)

```bash
python evaluate/evaluate_all.py
```

Outputs:

- `evaluate/output/<run>/metrics.json` for each run (next to `labelled_sentiment.csv`)

Optional cross-run summary:

```bash
python evaluate/evaluate_all.py --write-summary
```

This additionally writes:

- `evaluate/output/summary.csv`

## Script responsibilities (quick reference)

- `run_eval.py`: create a new scored run in `output/`
- `evaluate_one.py`: evaluate exactly one `labelled_sentiment.csv`
- `evaluate_all.py`: evaluate every run under `output/` and write per-run `metrics.json`
