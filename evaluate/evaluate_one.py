import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

VALID_LABELS = ["bearish", "neutral", "bullish"]


def compute_metrics(input_csv: str) -> dict:
    """
    Returns a JSON-serializable metrics dict for programmatic use.
    Terminal output should be handled by evaluate()/print_report().
    """
    df = pd.read_csv(input_csv)
    columns = df.columns.tolist()

    df = df.dropna(subset=["true_label", "label"]).copy()
    df["true_label"] = df["true_label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["true_label"].isin(VALID_LABELS) & df["label"].isin(VALID_LABELS)]
    y_true = df["true_label"]
    y_pred = df["label"]

    accuracy = float(accuracy_score(y_true, y_pred)) if len(df) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)

    return {
        "input_csv": str(input_csv),
        "columns": columns,
        "rows_evaluated": int(len(df)),
        "accuracy": accuracy,
        "labels": list(VALID_LABELS),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            digits=4,
            labels=VALID_LABELS,
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate(input_csv: str) -> None:
    """
    Terminal output intentionally mirrors the original `evaluate_score.py` format.
    """
    df = pd.read_csv(input_csv)

    # Check columns
    print("Columns:", df.columns.tolist())

    # Drop rows without labels
    df = df.dropna(subset=["true_label", "label"]).copy()

    # Normalize labels (VERY IMPORTANT)
    df["true_label"] = df["true_label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Optional: enforce valid labels only
    df = df[df["true_label"].isin(VALID_LABELS) & df["label"].isin(VALID_LABELS)]

    # Basic stats
    print("Rows evaluated:", len(df))
    print("Accuracy:", round(accuracy_score(df["true_label"], df["label"]), 4))
    print()

    # Full report
    print(
        classification_report(
            df["true_label"],
            df["label"],
            digits=4,
            labels=VALID_LABELS,
            zero_division=0,
        )
    )

    # Confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(df["true_label"], df["label"], labels=VALID_LABELS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a labelled sentiment CSV.")
    default_csv = Path(__file__).resolve().parent / "output" / "latest" / "labelled_sentiment.csv"
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(default_csv),
        help="Path to labelled CSV (must contain columns: label, true_label).",
    )
    args = parser.parse_args()
    evaluate(args.input_csv)


if __name__ == "__main__":
    main()

