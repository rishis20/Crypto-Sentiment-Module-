import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_one import compute_metrics, evaluate


def find_labelled_csvs(output_dir: Path) -> list[Path]:
    # Convention: evaluate/output/<model_slug>/<run_id>/labelled_sentiment.csv
    return sorted(output_dir.glob("*/*/*labelled_sentiment.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all runs in evaluate/output/."
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory (relative to evaluate/) that contains run folders.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Also write evaluate/output/summary.csv (cross-run aggregate).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labelled_csvs = find_labelled_csvs(output_dir)
    if not labelled_csvs:
        raise SystemExit(f"No labelled_sentiment.csv files found under: {output_dir}")

    rows: list[dict] = []
    for csv_path in labelled_csvs:
        metrics = compute_metrics(str(csv_path))
        run_dir = csv_path.parent
        run_name = f"{run_dir.parent.name}/{run_dir.name}"
        rows.append(
            {
                "run": run_name,
                "path": str(csv_path.relative_to(script_dir)),
                "rows_evaluated": metrics["rows_evaluated"],
                "accuracy": metrics["accuracy"],
            }
        )
        # Keep run-specific metrics right next to the labelled CSV.
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print()
        print("=" * 80)
        print(f"RUN: {run_name}")
        print(f"FILE: {csv_path}")
        print("=" * 80)
        evaluate(str(csv_path))

    summary_df = pd.DataFrame(rows).sort_values(["accuracy", "run"], ascending=[False, True])
    print(f"Evaluated {len(rows)} runs.")
    print(summary_df.to_string(index=False))

    if args.write_summary:
        summary_csv = output_dir / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()

