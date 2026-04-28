import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def evaluate(input_csv: str) -> None:
    df = pd.read_csv(input_csv)

    # Check columns
    print("Columns:", df.columns.tolist())

    # Drop rows without labels
    df = df.dropna(subset=["true_label", "label"]).copy()

    # Normalize labels (VERY IMPORTANT)
    df["true_label"] = df["true_label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Optional: enforce valid labels only
    valid_labels = ["bearish", "neutral", "bullish"]
    df = df[df["true_label"].isin(valid_labels) & df["label"].isin(valid_labels)]

    # Basic stats
    print("Rows evaluated:", len(df))
    print("Accuracy:", round(accuracy_score(df["true_label"], df["label"]), 4))
    print()

    # Full report
    print(classification_report(df["true_label"], df["label"], digits=4))

    # Confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(df["true_label"], df["label"], labels=valid_labels))


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment labels in a scored CSV.")
    default_csv = Path(__file__).resolve().parent / "labelled_sentiment_llama3.2.csv"
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(default_csv),
        help="Path to the labeled/scored CSV file (defaults to evaluate/labelled_sentiment_llama3.2.csv).",
    )
    args = parser.parse_args()
    evaluate(args.input_csv)


if __name__ == "__main__":
    main()
