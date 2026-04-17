"""
Render all cleaning experiment visualizations in one run.

Run:
    python3 visualize_all.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


def latest_stats_file() -> Path:
    stats_files = sorted(Path("data/reports").glob("cleaning_stats_*.json"))
    if not stats_files:
        raise FileNotFoundError("No stats files found in data/reports. Run pipeline first.")
    return stats_files[-1]


def latest_clean_file() -> Path:
    clean_files = sorted(Path("data/clean").glob("news_clean_*.jsonl"))
    if not clean_files:
        raise FileNotFoundError("No clean files found in data/clean. Run pipeline first.")
    return clean_files[-1]


def main() -> None:
    stats_path = latest_stats_file()
    clean_path = latest_clean_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    reasons = stats.get("reject_reason_counts", {})
    raw = stats.get("by_source", {}).get("raw", {})
    clean = stats.get("by_source", {}).get("clean", {})

    lengths = []
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lengths.append(len(row.get("text_for_model", "")))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Chart 1: reject reasons
    if reasons:
        axes[0].bar(list(reasons.keys()), [reasons[k] for k in reasons.keys()])
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_title("Reject reasons")
        axes[0].set_ylabel("count")
    else:
        axes[0].text(0.5, 0.5, "No reject data", ha="center", va="center")
        axes[0].set_title("Reject reasons")

    # Chart 2: raw vs clean by source
    sources = sorted(set(raw) | set(clean))
    if sources:
        x = np.arange(len(sources))
        width = 0.36
        axes[1].bar(x - width / 2, [raw.get(s, 0) for s in sources], width=width, label="raw")
        axes[1].bar(x + width / 2, [clean.get(s, 0) for s in sources], width=width, label="clean")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sources)
        axes[1].set_title("Raw vs Clean by source")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No source data", ha="center", va="center")
        axes[1].set_title("Raw vs Clean by source")

    # Chart 3: text lengths
    if lengths:
        axes[2].hist(lengths, bins=25)
        axes[2].set_title("Text length distribution")
        axes[2].set_xlabel("characters")
        axes[2].set_ylabel("count")
    else:
        axes[2].text(0.5, 0.5, "No clean data", ha="center", va="center")
        axes[2].set_title("Text length distribution")

    fig.suptitle(f"Cleaning experiment dashboard ({stats_path.name})", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
