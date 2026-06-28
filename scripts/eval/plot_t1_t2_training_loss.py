"""
Plot training losses from the local T1/T2 CycleGAN CSV log.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generator and discriminator losses from training_log.csv."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to outputs/runs/.../logs/training_log.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write the loss plot PNG.",
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

    steps = [int(row["step"]) for row in rows]
    loss_g = [float(row["loss_G"]) for row in rows]
    loss_d = [float(row["loss_D"]) for row in rows]
    loss_g_a2b = [float(row["loss_G_A2B"]) for row in rows]
    loss_g_b2a = [float(row["loss_G_B2A"]) for row in rows]
    loss_d_a2b = [float(row["loss_D_A2B"]) for row in rows]
    loss_d_b2a = [float(row["loss_D_B2A"]) for row in rows]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, loss_g, label="loss_G", linewidth=1.8)
    ax.plot(steps, loss_d, label="loss_D", linewidth=1.8)
    ax.plot(steps, loss_g_a2b, label="loss_G_A2B", linewidth=1.0, alpha=0.9)
    ax.plot(steps, loss_g_b2a, label="loss_G_B2A", linewidth=1.0, alpha=0.9)
    ax.plot(steps, loss_d_a2b, label="loss_D_A2B", linewidth=1.0, alpha=0.9)
    ax.plot(steps, loss_d_b2a, label="loss_D_B2A", linewidth=1.0, alpha=0.9)
    ax.set_title("T1/T2 CycleGAN Training Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=160)
    print(f"Wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
