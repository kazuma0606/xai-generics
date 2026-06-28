"""
Compute image-level metrics for paired generated MRI slice PNG files.

The current implementation is a reproducible reimplementation based on standard
Python libraries rather than a recovered legacy notebook cell.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from imageio import v2 as imageio


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    real_path: str
    fake_path: str


@dataclass(frozen=True)
class MetricRecord:
    pair_id: str
    real_path: str
    fake_path: str
    rmse: float
    mutual_information: float
    psnr: float
    ssim: float


def direction_suffixes(direction: str) -> tuple[str, str]:
    if direction == "AtoB":
        return "_real_B", "_fake_B"
    if direction == "BtoA":
        return "_real_A", "_fake_A"
    raise ValueError(f"Unsupported direction: {direction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RMSE, MI, PSNR, and SSIM for paired PNG outputs."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory containing paired image files such as *_real_B.png and *_fake_B.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write CSV metric outputs.",
    )
    parser.add_argument(
        "--direction",
        choices=["AtoB", "BtoA"],
        default="AtoB",
        help="CycleGAN generation direction used to choose the evaluation pair.",
    )
    parser.add_argument(
        "--real-suffix",
        default=None,
        help="Optional override for the reference-image filename stem suffix.",
    )
    parser.add_argument(
        "--fake-suffix",
        default=None,
        help="Optional override for the generated-image filename stem suffix.",
    )
    parser.add_argument(
        "--extension",
        default=".png",
        help="Filename extension to match.",
    )
    parser.add_argument(
        "--mi-bins",
        type=int,
        default=64,
        help="Number of grayscale bins used for mutual information.",
    )
    return parser.parse_args()


def require_scikit_image():
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:
        raise SystemExit(
            "scikit-image is required for PSNR and SSIM. "
            "Install project dependencies with uv before running this script."
        ) from exc
    return peak_signal_noise_ratio, structural_similarity


def require_scikit_learn():
    try:
        from sklearn.metrics import mutual_info_score
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for mutual information. "
            "Install project dependencies with uv before running this script."
        ) from exc
    return mutual_info_score


def pair_key(path: Path, suffix: str) -> str:
    stem = path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected filename stem for suffix {suffix}: {path.name}")
    return stem[: -len(suffix)]


def find_pairs(
    image_root: Path,
    real_suffix: str,
    fake_suffix: str,
    extension: str,
) -> list[PairRecord]:
    real_paths = sorted(image_root.glob(f"*{real_suffix}{extension}"))
    fake_lookup: dict[str, Path] = {}
    for fake_path in sorted(image_root.glob(f"*{fake_suffix}{extension}")):
        fake_lookup[pair_key(fake_path, fake_suffix)] = fake_path
    pairs: list[PairRecord] = []
    for real_path in real_paths:
        key = pair_key(real_path, real_suffix)
        fake_path = fake_lookup.get(key)
        if fake_path is None:
            continue
        pairs.append(
            PairRecord(pair_id=key, real_path=str(real_path), fake_path=str(fake_path))
        )
    return pairs


def load_image(path: str) -> np.ndarray:
    return np.asarray(imageio.imread(path), dtype=np.float32)


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        # Use luminance weighting so MI does not depend on raw RGB tuple counts.
        gray = (
            0.2989 * image[:, :, 0]
            + 0.5870 * image[:, :, 1]
            + 0.1140 * image[:, :, 2]
        )
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return np.clip(gray, 0, 255).astype(np.uint8)


def rmse(real: np.ndarray, fake: np.ndarray) -> float:
    diff = real.astype(np.float32) - fake.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))


def mutual_information(
    real: np.ndarray,
    fake: np.ndarray,
    bins: int,
    mutual_info_score,
) -> float:
    real_gray = to_grayscale_uint8(real)
    fake_gray = to_grayscale_uint8(fake)
    real_binned = np.floor(real_gray.astype(np.float32) * bins / 256.0).astype(np.int32)
    fake_binned = np.floor(fake_gray.astype(np.float32) * bins / 256.0).astype(np.int32)
    real_binned = np.clip(real_binned, 0, bins - 1)
    fake_binned = np.clip(fake_binned, 0, bins - 1)
    return float(
        mutual_info_score(real_binned.reshape(-1), fake_binned.reshape(-1))
    )


def psnr(real: np.ndarray, fake: np.ndarray, peak_signal_noise_ratio) -> float:
    return float(peak_signal_noise_ratio(real, fake, data_range=255.0))


def ssim(real: np.ndarray, fake: np.ndarray, structural_similarity) -> float:
    channel_axis = -1 if real.ndim == 3 else None
    return float(
        structural_similarity(real, fake, data_range=255.0, channel_axis=channel_axis)
    )


def compute_metrics(
    pairs: list[PairRecord],
    mi_bins: int,
) -> list[MetricRecord]:
    peak_signal_noise_ratio, structural_similarity = require_scikit_image()
    mutual_info_score = require_scikit_learn()
    records: list[MetricRecord] = []
    for pair in pairs:
        real = load_image(pair.real_path)
        fake = load_image(pair.fake_path)
        if real.shape != fake.shape:
            raise ValueError(
                f"Shape mismatch for {pair.pair_id}: {real.shape} vs {fake.shape}"
            )
        records.append(
            MetricRecord(
                pair_id=pair.pair_id,
                real_path=pair.real_path,
                fake_path=pair.fake_path,
                rmse=rmse(real, fake),
                mutual_information=mutual_information(
                    real, fake, bins=mi_bins, mutual_info_score=mutual_info_score
                ),
                psnr=psnr(real, fake, peak_signal_noise_ratio),
                ssim=ssim(real, fake, structural_similarity),
            )
        )
    return records


def write_per_pair_csv(output_path: Path, records: list[MetricRecord]) -> None:
    fieldnames = [
        "pair_id",
        "real_path",
        "fake_path",
        "rmse",
        "mutual_information",
        "psnr",
        "ssim",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def summarize(records: list[MetricRecord]) -> dict[str, float]:
    def mean(values: list[float]) -> float:
        if not values:
            return math.nan
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    return {
        "pair_count": float(len(records)),
        "rmse_mean": mean([record.rmse for record in records]),
        "mutual_information_mean": mean(
            [record.mutual_information for record in records]
        ),
        "psnr_mean": mean([record.psnr for record in records]),
        "ssim_mean": mean([record.ssim for record in records]),
    }


def write_summary_csv(output_path: Path, summary: dict[str, float]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_real_suffix, default_fake_suffix = direction_suffixes(args.direction)
    real_suffix = args.real_suffix or default_real_suffix
    fake_suffix = args.fake_suffix or default_fake_suffix
    pairs = find_pairs(
        image_root=args.image_root,
        real_suffix=real_suffix,
        fake_suffix=fake_suffix,
        extension=args.extension,
    )
    if not pairs:
        raise SystemExit(
            f"No paired images found under {args.image_root} "
            f"for direction={args.direction} ({real_suffix}, {fake_suffix})"
        )
    records = compute_metrics(pairs, mi_bins=args.mi_bins)
    per_pair_path = args.output_dir / "per_pair_metrics.csv"
    summary_path = args.output_dir / "summary_metrics.csv"
    write_per_pair_csv(per_pair_path, records)
    write_summary_csv(summary_path, summarize(records))
    print(f"Pairs evaluated: {len(records)}")
    print(f"Direction: {args.direction}")
    print(f"Reference suffix: {real_suffix}")
    print(f"Generated suffix: {fake_suffix}")
    print(f"Wrote per-pair metrics: {per_pair_path}")
    print(f"Wrote summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
