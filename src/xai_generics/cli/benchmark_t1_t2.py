from __future__ import annotations

import argparse
import time
from pathlib import Path

from xai_generics.config import load_yaml
from xai_generics.data.t1_t2_pairs import T1T2PairDataset
from xai_generics.models.cyclegan import require_torch
from xai_generics.training.cyclegan_trainer import CycleGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local T1/T2 runtime.")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/2d/train_t1_t2_cyclegan.yaml"),
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/2d/dataset_t1_t2.yaml"),
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="Benchmark a training step or an inference step.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of timed steps to measure.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed warmup steps.",
    )
    parser.add_argument(
        "--direction",
        choices=["AtoB", "BtoA"],
        default="AtoB",
        help="Direction used for inference benchmarking.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["training", "generation_test"],
        default=None,
        help="Raw-root split used when no manifest is provided.",
    )
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace) -> T1T2PairDataset:
    dataset_config = load_yaml(args.dataset_config)
    if args.train_manifest is not None:
        return T1T2PairDataset.from_manifest(args.train_manifest)
    if args.dataset_split is None:
        split = "training" if args.mode == "train" else "generation_test"
    else:
        split = args.dataset_split
    split_key = "training_root" if split == "training" else "generation_test_root"
    preprocessing = dataset_config.get("preprocessing", {})
    resize_to = None
    if isinstance(preprocessing, dict):
        resize_cfg = preprocessing.get("resize", {})
        if isinstance(resize_cfg, dict) and resize_cfg.get("enabled", False):
            load_size = resize_cfg.get("load_size")
            if load_size is not None:
                resize_to = int(load_size)
    return T1T2PairDataset.from_raw_root(
        Path(dataset_config["data"][split_key]),
        source=dataset_config["modalities"]["source"],
        target=dataset_config["modalities"]["target"],
        resize_to=resize_to,
    )


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset = _load_dataset(args)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty.")
    trainer = CycleGANTrainer.from_config(train_config)
    torch, _, _ = require_torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    sample_count = min(args.warmup + args.steps, len(dataset))
    timings: list[float] = []
    for index in range(sample_count):
        sample = dataset[index]
        start = time.perf_counter()
        if args.mode == "train":
            trainer.train_step(sample["source"], sample["target"])
        else:
            if args.direction == "AtoB":
                trainer.infer_AtoB(sample["source"])
            else:
                trainer.infer_BtoA(sample["target"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index >= args.warmup:
            timings.append(elapsed)
    if not timings:
        raise SystemExit("No timed steps were collected.")
    avg_step = sum(timings) / len(timings)
    estimated_epoch_seconds = avg_step * len(dataset)
    estimated_200_epoch_seconds = estimated_epoch_seconds * 200.0
    print(f"mode={args.mode}")
    print(f"samples={len(dataset)}")
    print(f"timed_steps={len(timings)}")
    print(f"avg_step_seconds={avg_step:.4f}")
    print(f"estimated_epoch_seconds={estimated_epoch_seconds:.2f}")
    print(f"estimated_200_epoch_hours={estimated_200_epoch_seconds / 3600.0:.2f}")
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
        print(f"peak_cuda_allocated_mb={peak_allocated:.2f}")
        print(f"peak_cuda_reserved_mb={peak_reserved:.2f}")
    else:
        print("peak_cuda_allocated_mb=n/a")
        print("peak_cuda_reserved_mb=n/a")


if __name__ == "__main__":
    main()
