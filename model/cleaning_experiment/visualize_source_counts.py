"""
Plot raw vs clean record counts by source from latest stats.

Run:
    python3 visualize_source_counts.py
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


def main() -> None:
    stats_path = latest_stats_file()
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    raw = data.get("by_source", {}).get("raw", {})
    clean = data.get("by_source", {}).get("clean", {})

    sources = sorted(set(raw) | set(clean))
    if not sources:
        print("No source counts found in stats file.")
        return

    raw_vals = [raw.get(s, 0) for s in sources]
    clean_vals = [clean.get(s, 0) for s in sources]

    x = np.arange(len(sources))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, raw_vals, width=width, label="raw")
    plt.bar(x + width / 2, clean_vals, width=width, label="clean")
    plt.xticks(x, sources)
    plt.ylabel("Record count")
    plt.title(f"Raw vs Clean by source ({stats_path.name})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
