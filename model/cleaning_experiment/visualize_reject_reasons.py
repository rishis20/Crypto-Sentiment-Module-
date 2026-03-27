"""
Plot reject reason counts from the latest cleaning stats file.

Run:
    python3 visualize_reject_reasons.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt


def latest_stats_file() -> Path:
    stats_files = sorted(Path("data/reports").glob("cleaning_stats_*.json"))
    if not stats_files:
        raise FileNotFoundError("No stats files found in data/reports. Run pipeline first.")
    return stats_files[-1]


def main() -> None:
    stats_path = latest_stats_file()
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    reasons = data.get("reject_reason_counts", {})
    if not reasons:
        print("No reject reasons found in stats file.")
        return

    labels = list(reasons.keys())
    values = [reasons[k] for k in labels]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Rejected records")
    plt.title(f"Reject reasons ({stats_path.name})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
