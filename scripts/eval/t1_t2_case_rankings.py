"""
Rank cases from an existing case_summary_metrics.csv file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create worst/best case rankings from case summary metrics."
    )
    parser.add_argument("--case-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str) -> float:
    return float(value)


def top_rows(
    rows: list[dict[str, str]],
    metric: str,
    reverse: bool,
    top_k: int,
) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda row: numeric(row[metric]), reverse=reverse)
    return ranked[:top_k]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    rmse_rows: list[dict[str, str]],
    mi_rows: list[dict[str, str]],
    ssim_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Case Rankings",
        "",
        "## Highest RMSE",
        "",
        "| rank | case_id | rmse_mean | ssim_mean | mutual_information_mean |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(rmse_rows, start=1):
        lines.append(
            f"| {idx} | {row['case_id']} | {row['rmse_mean']} | {row['ssim_mean']} | {row['mutual_information_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Lowest Mutual Information",
            "",
            "| rank | case_id | mutual_information_mean | rmse_mean | ssim_mean |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(mi_rows, start=1):
        lines.append(
            f"| {idx} | {row['case_id']} | {row['mutual_information_mean']} | {row['rmse_mean']} | {row['ssim_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Lowest SSIM",
            "",
            "| rank | case_id | ssim_mean | rmse_mean | mutual_information_mean |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(ssim_rows, start=1):
        lines.append(
            f"| {idx} | {row['case_id']} | {row['ssim_mean']} | {row['rmse_mean']} | {row['mutual_information_mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.case_summary_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.case_summary_csv}")
    rmse_rows = top_rows(rows, "rmse_mean", reverse=True, top_k=args.top_k)
    mi_rows = top_rows(
        rows, "mutual_information_mean", reverse=False, top_k=args.top_k
    )
    ssim_rows = top_rows(rows, "ssim_mean", reverse=False, top_k=args.top_k)
    write_csv(args.output_dir / "top_rmse_cases.csv", rmse_rows)
    write_csv(args.output_dir / "bottom_mi_cases.csv", mi_rows)
    write_csv(args.output_dir / "bottom_ssim_cases.csv", ssim_rows)
    write_markdown(
        args.output_dir / "case_rankings.md",
        rmse_rows=rmse_rows,
        mi_rows=mi_rows,
        ssim_rows=ssim_rows,
    )
    print(f"Rows ranked: {len(rows)}")
    print(f"Wrote rankings under: {args.output_dir}")


if __name__ == "__main__":
    main()
