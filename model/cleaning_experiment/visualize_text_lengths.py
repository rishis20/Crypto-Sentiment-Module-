"""
Plot text length distribution from latest clean JSONL output.

Run:
    python3 visualize_text_lengths.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt


def latest_clean_file() -> Path:
    clean_files = sorted(Path("data/clean").glob("news_clean_*.jsonl"))
    if not clean_files:
        raise FileNotFoundError("No clean files found in data/clean. Run pipeline first.")
    return clean_files[-1]


def main() -> None:
    clean_path = latest_clean_file()
    lengths = []
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lengths.append(len(row.get("text_for_model", "")))

    if not lengths:
        print("No text lengths found in clean file.")
        return

    plt.figure(figsize=(9, 5))
    plt.hist(lengths, bins=25)
    plt.xlabel("text_for_model length (characters)")
    plt.ylabel("Count")
    plt.title(f"Text length distribution ({clean_path.name})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
