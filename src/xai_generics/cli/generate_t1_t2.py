from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from xai_generics.config import load_yaml
from xai_generics.data.t1_t2_pairs import T1T2PairDataset
from xai_generics.training.cyclegan_trainer import CycleGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired T1/T2 outputs for the generation-side test set."
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/runs/t1_t2_cyclegan_2d/checkpoints"),
    )
    parser.add_argument("--checkpoint-tag", default="latest")
    parser.add_argument(
        "--dataset-split",
        choices=["training", "generation_test"],
        default="generation_test",
        help="Raw-root split to read when no manifest is provided.",
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generated/t1_t2_cyclegan_2d/generation_test"),
    )
    parser.add_argument(
        "--direction",
        choices=["AtoB", "BtoA"],
        default="AtoB",
        help="Generation direction used for saving real/fake pairs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of paired slices to generate.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=200,
        help="Progress print cadence in generated pairs.",
    )
    return parser.parse_args()


def _tensor_to_uint8_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor output, got {image.shape}")
    image = np.clip((image + 1.0) * 127.5, 0.0, 255.0)
    image = np.rint(image).astype(np.uint8)
    return np.transpose(image, (1, 2, 0))


def _slice_to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got {image.shape}")
    image = np.clip(image, 0.0, 255.0)
    image = np.rint(image).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2)


def _load_dataset(dataset_config: dict[str, object], args: argparse.Namespace) -> T1T2PairDataset:
    if args.train_manifest is not None:
        return T1T2PairDataset.from_manifest(args.train_manifest)
    split_key = "training_root" if args.dataset_split == "training" else "generation_test_root"
    data_cfg = dataset_config["data"]
    modalities = dataset_config["modalities"]
    preprocessing = dataset_config.get("preprocessing", {})
    resize_to = None
    if isinstance(preprocessing, dict):
        resize_cfg = preprocessing.get("resize", {})
        if isinstance(resize_cfg, dict) and resize_cfg.get("enabled", False):
            load_size = resize_cfg.get("load_size")
            if load_size is not None:
                resize_to = int(load_size)
    if not isinstance(data_cfg, dict) or not isinstance(modalities, dict):
        raise ValueError("Invalid dataset config structure.")
    return T1T2PairDataset.from_raw_root(
        Path(data_cfg[split_key]),
        source=str(modalities["source"]),
        target=str(modalities["target"]),
        resize_to=resize_to,
    )


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset_config = load_yaml(args.dataset_config)
    dataset = _load_dataset(dataset_config, args)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty.")
    trainer = CycleGANTrainer.from_config(train_config)
    trainer.bundle.load_checkpoint(args.checkpoint_dir, args.checkpoint_tag, strict=True)
    if args.direction == "AtoB":
        real_kind = "real_B"
        fake_kind = "fake_B"
    else:
        real_kind = "real_A"
        fake_kind = "fake_A"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    rows: list[dict[str, object]] = []
    for index in range(limit):
        sample = dataset[index]
        if args.direction == "AtoB":
            fake = trainer.infer_AtoB(sample["source"])
            real_image = _slice_to_rgb_uint8(sample["target"])
        else:
            fake = trainer.infer_BtoA(sample["target"])
            real_image = _slice_to_rgb_uint8(sample["source"])
        fake_image = _tensor_to_uint8_image(fake)
        prefix = f"{sample['patient_id']}_{int(sample['slice_index'])}"
        real_path = args.output_dir / f"{prefix}_{real_kind}.png"
        fake_path = args.output_dir / f"{prefix}_{fake_kind}.png"
        imageio.imwrite(real_path, real_image)
        imageio.imwrite(fake_path, fake_image)
        rows.append(
            {
                "pair_id": prefix,
                "patient_id": sample["patient_id"],
                "slice_index": sample["slice_index"],
                "direction": args.direction,
                "real_kind": real_kind,
                "fake_kind": fake_kind,
                "real_path": str(real_path),
                "fake_path": str(fake_path),
            }
        )
        if args.print_every > 0 and (index + 1) % args.print_every == 0:
            print(f"generated={index + 1}/{limit}")
    manifest_path = args.output_dir / "generation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Pairs available: {len(dataset)}")
    print(f"Generated pairs: {len(rows)}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
