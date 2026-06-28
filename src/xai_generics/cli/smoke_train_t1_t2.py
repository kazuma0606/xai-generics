from __future__ import annotations

import argparse
from pathlib import Path

from xai_generics.config import load_yaml
from xai_generics.data.t1_t2_pairs import T1T2PairDataset
from xai_generics.training.cyclegan_trainer import CycleGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single local CycleGAN smoke step for T1/T2."
    )
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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/runs/t1_t2_cyclegan_2d/checkpoints"))
    parser.add_argument("--checkpoint-tag", default="smoke")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset loading without instantiating PyTorch models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset_config = load_yaml(args.dataset_config)
    if args.train_manifest is not None:
        dataset = T1T2PairDataset.from_manifest(args.train_manifest)
    else:
        dataset = T1T2PairDataset.from_raw_root(
            Path(dataset_config["data"]["training_root"]),
            source=dataset_config["modalities"]["source"],
            target=dataset_config["modalities"]["target"],
        )
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty.")
    if args.dry_run:
        first = dataset[0]
        print(
            f"Dry run completed. First sample: patient={first['patient_id']} "
            f"slice={first['slice_index']} source_shape={first['source'].shape} "
            f"target_shape={first['target'].shape}"
        )
        return
    trainer = CycleGANTrainer.from_config(train_config)
    print(f"Pairs available: {len(dataset)}")
    for index in range(min(args.limit, len(dataset))):
        sample = dataset[index]
        metrics = trainer.train_step(sample["source"], sample["target"])
        print(
            f"step={index} patient={sample['patient_id']} slice={sample['slice_index']} "
            f"loss_G={metrics['loss_G']:.4f} loss_D={metrics['loss_D']:.4f}"
        )
    trainer.save(args.checkpoint_dir, args.checkpoint_tag)
    print(f"Checkpoint saved under: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
