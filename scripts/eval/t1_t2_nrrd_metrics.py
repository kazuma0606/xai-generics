"""
Compute image metrics directly from reconstructed NRRD volumes.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import nrrd
import numpy as np


@dataclass(frozen=True)
class SliceMetricRecord:
    slice_index: int
    rmse: float
    mutual_information: float
    psnr: float
    ssim: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RMSE, MI, PSNR, and SSIM from paired NRRD volumes."
    )
    parser.add_argument("--real-volume", type=Path, required=True)
    parser.add_argument("--fake-volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mi-bins", type=int, default=64)
    return parser.parse_args()


def require_scikit_image():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    return peak_signal_noise_ratio, structural_similarity


def require_scikit_learn():
    from sklearn.metrics import mutual_info_score

    return mutual_info_score


def load_volume(path: Path) -> np.ndarray:
    volume, _ = nrrd.read(str(path))
    return np.asarray(volume, dtype=np.float32)


def rmse(real: np.ndarray, fake: np.ndarray) -> float:
    diff = real - fake
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))


def mutual_information(
    real: np.ndarray,
    fake: np.ndarray,
    bins: int,
    mutual_info_score,
) -> float:
    real_binned = np.floor(real.astype(np.float32) * bins / 256.0).astype(np.int32)
    fake_binned = np.floor(fake.astype(np.float32) * bins / 256.0).astype(np.int32)
    real_binned = np.clip(real_binned, 0, bins - 1)
    fake_binned = np.clip(fake_binned, 0, bins - 1)
    return float(mutual_info_score(real_binned.reshape(-1), fake_binned.reshape(-1)))


def write_slice_csv(path: Path, records: list[SliceMetricRecord]) -> None:
    fieldnames = ["slice_index", "rmse", "mutual_information", "psnr", "ssim"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_summary_csv(path: Path, summary: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def summarize(records: list[SliceMetricRecord]) -> dict[str, float]:
    def mean(values: list[float]) -> float:
        if not values:
            return math.nan
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def finite_mean(values: list[float]) -> float:
        if not values:
            return math.nan
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            return math.inf
        return float(np.mean(np.asarray(finite, dtype=np.float64)))

    psnr_values = [record.psnr for record in records]

    return {
        "slice_count": float(len(records)),
        "rmse_mean": mean([record.rmse for record in records]),
        "mutual_information_mean": mean(
            [record.mutual_information for record in records]
        ),
        "psnr_mean": mean(psnr_values),
        "psnr_mean_finite_only": finite_mean(psnr_values),
        "psnr_infinite_slice_count": float(sum(not np.isfinite(value) for value in psnr_values)),
        "ssim_mean": mean([record.ssim for record in records]),
    }


def main() -> None:
    args = parse_args()
    peak_signal_noise_ratio, structural_similarity = require_scikit_image()
    mutual_info_score = require_scikit_learn()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = load_volume(args.real_volume)
    fake = load_volume(args.fake_volume)
    if real.shape != fake.shape:
        raise SystemExit(f"Shape mismatch: {real.shape} vs {fake.shape}")
    if real.ndim != 3:
        raise SystemExit(f"Expected 3D volumes, got {real.shape}")
    records: list[SliceMetricRecord] = []
    for index in range(real.shape[0]):
        real_slice = real[index]
        fake_slice = fake[index]
        mse = float(np.mean(np.square(real_slice - fake_slice), dtype=np.float64))
        psnr_value = (
            math.inf
            if mse == 0.0
            else float(peak_signal_noise_ratio(real_slice, fake_slice, data_range=255.0))
        )
        records.append(
            SliceMetricRecord(
                slice_index=index,
                rmse=rmse(real_slice, fake_slice),
                mutual_information=mutual_information(
                    real_slice, fake_slice, bins=args.mi_bins, mutual_info_score=mutual_info_score
                ),
                psnr=psnr_value,
                ssim=float(structural_similarity(real_slice, fake_slice, data_range=255.0)),
            )
        )
    slice_csv = args.output_dir / "per_slice_metrics.csv"
    summary_csv = args.output_dir / "summary_metrics.csv"
    write_slice_csv(slice_csv, records)
    write_summary_csv(summary_csv, summarize(records))
    print(f"Slices evaluated: {len(records)}")
    print(f"Wrote per-slice metrics: {slice_csv}")
    print(f"Wrote summary metrics: {summary_csv}")


if __name__ == "__main__":
    main()
