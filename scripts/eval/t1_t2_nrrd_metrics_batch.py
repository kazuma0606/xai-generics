"""
Batch-evaluate paired real/fake T2 NRRD volumes across case directories.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from t1_t2_nrrd_metrics import (
    SliceMetricRecord,
    load_volume,
    mutual_information,
    require_scikit_image,
    require_scikit_learn,
    rmse,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-compute RMSE, MI, PSNR, and SSIM across case NRRD pairs."
    )
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-name", default="real_T2.nrrd")
    parser.add_argument("--fake-name", default="fake_T2_epoch_200.nrrd")
    parser.add_argument("--mi-bins", type=int, default=64)
    parser.add_argument("--limit-cases", type=int, default=None)
    return parser.parse_args()


def case_dirs(cases_root: Path) -> list[Path]:
    dirs = [path for path in cases_root.iterdir() if path.is_dir()]
    dirs.sort()
    return dirs


def slice_metrics_for_case(
    real_volume: np.ndarray,
    fake_volume: np.ndarray,
    mi_bins: int,
) -> list[SliceMetricRecord]:
    peak_signal_noise_ratio, structural_similarity = require_scikit_image()
    mutual_info_score = require_scikit_learn()
    records: list[SliceMetricRecord] = []
    for index in range(real_volume.shape[0]):
        real_slice = real_volume[index]
        fake_slice = fake_volume[index]
        mse = float(np.mean(np.square(real_slice - fake_slice), dtype=np.float64))
        psnr_value = (
            float("inf")
            if mse == 0.0
            else float(
                peak_signal_noise_ratio(real_slice, fake_slice, data_range=255.0)
            )
        )
        records.append(
            SliceMetricRecord(
                slice_index=index,
                rmse=rmse(real_slice, fake_slice),
                mutual_information=mutual_information(
                    real_slice,
                    fake_slice,
                    bins=mi_bins,
                    mutual_info_score=mutual_info_score,
                ),
                psnr=psnr_value,
                ssim=float(
                    structural_similarity(real_slice, fake_slice, data_range=255.0)
                ),
            )
        )
    return records


def write_case_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "case_id",
        "slice_count",
        "rmse_mean",
        "mutual_information_mean",
        "psnr_mean",
        "psnr_mean_finite_only",
        "psnr_infinite_slice_count",
        "ssim_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_overall_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    numeric_keys = [
        "slice_count",
        "rmse_mean",
        "mutual_information_mean",
        "psnr_mean_finite_only",
        "psnr_infinite_slice_count",
        "ssim_mean",
    ]
    overall = {"case_count": float(len(rows))}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        overall[f"{key}_mean_across_cases"] = float(
            np.mean(np.asarray(values, dtype=np.float64))
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall.keys()))
        writer.writeheader()
        writer.writerow(overall)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = case_dirs(args.cases_root)
    if args.limit_cases is not None:
        roots = roots[: args.limit_cases]
    rows: list[dict[str, float | str]] = []
    for case_root in roots:
        real_path = case_root / args.real_name
        fake_path = case_root / args.fake_name
        if not real_path.exists() or not fake_path.exists():
            continue
        real_volume = load_volume(real_path)
        fake_volume = load_volume(fake_path)
        if real_volume.shape != fake_volume.shape or real_volume.ndim != 3:
            raise SystemExit(
                f"Invalid case volume shape for {case_root.name}: "
                f"{real_volume.shape} vs {fake_volume.shape}"
            )
        summary = summarize(slice_metrics_for_case(real_volume, fake_volume, args.mi_bins))
        row: dict[str, float | str] = {"case_id": case_root.name}
        row.update(summary)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No valid case pairs found under {args.cases_root}")
    case_summary_path = args.output_dir / "case_summary_metrics.csv"
    overall_summary_path = args.output_dir / "overall_summary_metrics.csv"
    write_case_summary_csv(case_summary_path, rows)
    write_overall_summary_csv(overall_summary_path, rows)
    print(f"Cases evaluated: {len(rows)}")
    print(f"Wrote case summary: {case_summary_path}")
    print(f"Wrote overall summary: {overall_summary_path}")


if __name__ == "__main__":
    main()
