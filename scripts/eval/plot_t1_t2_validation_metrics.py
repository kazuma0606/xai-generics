"""
Plot validation metrics from epoch-wise generation-side evaluation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation RMSE, MI, PSNR, and SSIM against training step."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to validation_log.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write the validation plot PNG.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.log_path)
    if not rows:
        raise SystemExit(f"No rows found in {args.log_path}")
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install project dependencies first."
        ) from exc

    steps = [int(float(row["step"])) for row in rows]
    rmse_values = [float(row["rmse_mean"]) for row in rows]
    mi_values = [float(row["mutual_information_mean"]) for row in rows]
    psnr_values = [float(row["psnr_mean"]) for row in rows]
    ssim_values = [float(row["ssim_mean"]) for row in rows]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    series = [
        ("RMSE", rmse_values),
        ("Mutual Information", mi_values),
        ("PSNR", psnr_values),
        ("SSIM", ssim_values),
    ]
    for ax, (title, values) in zip(axes.flat, series, strict=True):
        ax.plot(steps, values, linewidth=1.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.suptitle("Validation Metrics Over Training Step")
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=160)
    print(f"Wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
