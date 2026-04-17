# Cleaning Experiment (Small RSS Feed Set)

This directory contains a controlled preprocessing experiment for crypto-news sentiment.
It uses a small subset of RSS feeds and writes raw/clean/rejected artifacts separately for reproducibility.

## Goals

- Build a repeatable preprocessing baseline before scaling to all feeds.
- Keep raw vs derived data strictly separated.
- Improve LLM input quality by removing noise and invalid records.
- Log reject reasons and per-source metrics for analysis/reporting.

## Feed scope (pilot)

Defined in `pilot_config.py`:

- Reddit: `r/CryptoCurrency` RSS
- CoinDesk: consolidated RSS (`arc/outboundfeeds/rss`)
- CoinTelegraph: main RSS

This smaller set is intentional so preprocessing behavior is easier to inspect and tune.

## Pipeline script

- `run_small_feed_pipeline.py`

Run from this directory:

```bash
python3 run_small_feed_pipeline.py
```

## Output artifacts

The script writes:

- `data/raw/news_raw_YYYY-MM-DD.jsonl`
  - Immutable raw snapshot per fetched item.
- `data/clean/news_clean_YYYY-MM-DD.jsonl`
  - Cleaned records that pass validation and are ready for model scoring.
- `data/clean/news_rejected_YYYY-MM-DD.jsonl`
  - Rejected records with `reject_reason`.
- `data/scored/sentiment_YYYY-MM-DD.jsonl`
  - Ollama-scored records with `score`, `label`, `confidence`, `explanation`.
- `data/scored/sentiment_YYYY-MM-DD.csv`
  - CSV export of scored JSONL for comparison/reporting.
- `data/reports/cleaning_stats_YYYY-MM-DD.json`
  - Summary counts by source and reject reason.

## Preprocessing steps and justifications

### 1) Fetch robustness

Technique:
- HTTP fetch with explicit browser-like `User-Agent` (`requests`), then parse with `feedparser`.
- Fallback to `feedparser.parse(url)` if request path fails.

Why:
- Some providers return empty/blocked content with default parser user agents.
- Improves feed reliability across sources.

### 2) Raw snapshot preservation

Technique:
- Store `title_raw`, `summary_raw`, source metadata, canonicalized URL fields, timestamps, and raw payload.

Why:
- Supports reproducibility and debugging.
- Prevents irreversible information loss before cleaning decisions.

### 3) Universal text normalization

Technique:
- HTML stripping via BeautifulSoup.
- Whitespace collapse.
- Boilerplate text removal (`submitted by`, `[link]`, `[comments]`, etc.).

Why:
- RSS summaries often include platform wrappers rather than semantic text.
- Removes frequent artifacts that can bias or distract LLM scoring.

### 4) Source-specific filtering (Reddit-focused)

Technique:
- Reject thread-style posts (daily/weekly/monthly discussion patterns).
- Reject image/link-only content with little alphabetic text.

Why:
- Reddit contains many meta/discussion threads and media-only posts that are not article-like sentiment signals.
- Improves signal-to-noise ratio for market sentiment use cases.

### 5) Temporal and content validation

Technique:
- Date window filter (`MAX_DAYS_OLD`).
- Minimum content length (`MIN_CHAR_COUNT`).
- Language/garbled heuristic (`MIN_ALPHA_RATIO` + English hint words).

Why:
- Keeps corpus timely and meaningful.
- Prevents short/noisy records from destabilizing outputs.
- Routes likely unsupported language/encoding noise out of the scoring path.

### 6) Crypto relevance filtering

Technique:
- Keyword-based crypto detection on cleaned text (`CRYPTO_KEYWORDS`).

Why:
- Ensures kept records are in-domain for downstream crypto sentiment analysis.

### 7) Deterministic deduplication

Technique:
- Stable SHA-256 hash over normalized title+summary.
- Deduplicate both within-run and across previous clean runs.

Why:
- Avoids duplicate influence in downstream sentiment.
- Stable hashing is reproducible; unlike Python `hash(...)`, results persist across sessions.

### 8) Reject reason logging

Technique:
- Every filtered record gets a reason code in `news_rejected_*.jsonl`.

Common reasons:
- `invalid_date`
- `outside_date_window`
- `boilerplate_thread`
- `image_or_link_only`
- `too_short`
- `non_english_or_garbled`
- `non_crypto_after_clean`
- `duplicate`

Why:
- Makes preprocessing transparent and auditable.
- Enables targeted rule tuning instead of guesswork.

### 9) Scoring contract in experiment pipeline

Technique:
- Score each clean record using Ollama via parent `analyze.py` function (`analyze_text_sentiment`).
- Retry once if score is malformed/out-of-range.
- Enforce final score contract: clamp to `[-1, 1]`.
- Derive:
  - `label` (`bearish`, `neutral`, `bullish`) using thresholds
  - `confidence` from score magnitude
  - short `explanation`
- Persist scoring metadata (`model_name`, `prompt_version`, `scored_at`).

Why:
- Ensures the experiment pipeline is directly testable end-to-end against the production scoring approach.
- Keeps score schema stable for evaluation and report generation.

## Source-by-source justification summary

### Reddit

Observed issues:
- Heavy wrappers (`submitted by`, links/comments tags), recurring daily thread boilerplate, image/meme posts.

Applied choices:
- Stronger filtering for thread/meta and low-information posts.
- Keep only substantive text likely to carry market sentiment.

Rationale:
- Reddit has high volume but high noise variance; stricter cleaning improves precision.

### CoinDesk

Observed issues:
- Cleaner editorial feed overall; less social metadata noise.

Applied choices:
- Light boilerplate stripping + standard normalization.
- Keep narrative and numeric context intact.

Rationale:
- Over-cleaning can remove useful financial cues (percentages, tickers, action verbs).

### CoinTelegraph

Observed issues:
- Generally structured, but can include feed tails and repetitive formatting.

Applied choices:
- Standard cleanup and validation gates; no aggressive source-specific stripping yet.

Rationale:
- Similar to CoinDesk, quality is already moderate/high; preserve semantic content.

## Model input contract (clean stage)

Current canonical model input:

```text
title_clean

summary_clean
```

This format preserves headline salience and context without over-structuring.

## Sample visualization instructions

Use the stats file and clean/rejected JSONL outputs to create quick diagnostics.

### A) Reject reason bar chart

```bash
python3 - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt

stats_files = sorted(Path("data/reports").glob("cleaning_stats_*.json"))
latest = stats_files[-1]
data = json.loads(latest.read_text())
reasons = data["reject_reason_counts"]

plt.figure(figsize=(10,4))
plt.bar(reasons.keys(), reasons.values())
plt.xticks(rotation=35, ha="right")
plt.title("Rejected records by reason")
plt.tight_layout()
plt.show()
PY
```

### B) Raw vs clean count by source

```bash
python3 - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

stats_files = sorted(Path("data/reports").glob("cleaning_stats_*.json"))
latest = stats_files[-1]
data = json.loads(latest.read_text())

raw = data["by_source"]["raw"]
clean = data["by_source"]["clean"]
sources = sorted(set(raw) | set(clean))
raw_vals = [raw.get(s, 0) for s in sources]
clean_vals = [clean.get(s, 0) for s in sources]

x = np.arange(len(sources))
w = 0.35
plt.figure(figsize=(8,4))
plt.bar(x - w/2, raw_vals, width=w, label="raw")
plt.bar(x + w/2, clean_vals, width=w, label="clean")
plt.xticks(x, sources)
plt.title("Raw vs Clean records by source")
plt.legend()
plt.tight_layout()
plt.show()
PY
```

### C) Text length distribution (clean set)

```bash
python3 - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt

clean_files = sorted(Path("data/clean").glob("news_clean_*.jsonl"))
latest = clean_files[-1]
lengths = []
for line in latest.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    lengths.append(len(row.get("text_for_model", "")))

plt.figure(figsize=(8,4))
plt.hist(lengths, bins=20)
plt.title("Distribution of text_for_model length (chars)")
plt.xlabel("characters")
plt.ylabel("count")
plt.tight_layout()
plt.show()
PY
```

### D) Score distribution (scored set)

```bash
python3 - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt

scored_files = sorted(Path("data/scored").glob("sentiment_*.jsonl"))
latest = scored_files[-1]
scores = []
for line in latest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    scores.append(float(row.get("score", 0.0)))

plt.figure(figsize=(8,4))
plt.hist(scores, bins=25)
plt.title("Score distribution (Ollama)")
plt.xlabel("score")
plt.ylabel("count")
plt.tight_layout()
plt.show()
PY
```

## Suggested next iteration

After reviewing the charts and reject logs:

1. Tune per-source boilerplate patterns.
2. Add deterministic URL canonicalization rules per source.
3. Introduce optional language detection library (if needed).
4. Re-run and compare `cleaning_stats_*.json` across iterations.

