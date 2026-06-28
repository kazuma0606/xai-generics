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
        description="Local T1/T2 inference entrypoint scaffold."
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/2d/train_t1_t2_cyclegan.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/runs/t1_t2_cyclegan_2d/checkpoints"),
    )
    parser.add_argument(
        "--checkpoint-tag",
        default="latest",
        help="Checkpoint tag prefix, for example latest or 200.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and checkpoint naming without executing inference.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/2d/dataset_t1_t2.yaml"),
    )
    parser.add_argument(
        "--dataset-split",
        choices=["training", "generation_test"],
        default="generation_test",
        help="Which raw root to read when no manifest is provided.",
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
        default=8,
        help="Maximum number of paired slices to generate.",
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


def _save_pair(
    output_dir: Path,
    patient_id: str,
    slice_index: int,
    real_kind: str,
    fake_kind: str,
    real_image: np.ndarray,
    fake_image: np.ndarray,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{patient_id}_{slice_index}"
    real_path = output_dir / f"{prefix}_{real_kind}.png"
    fake_path = output_dir / f"{prefix}_{fake_kind}.png"
    imageio.imwrite(real_path, real_image)
    imageio.imwrite(fake_path, fake_image)
    return real_path, fake_path


def _resize_to(dataset_config: dict[str, object]) -> int | None:
    preprocessing = dataset_config.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        return None
    resize_cfg = preprocessing.get("resize", {})
    if not isinstance(resize_cfg, dict):
        return None
    if not resize_cfg.get("enabled", False):
        return None
    load_size = resize_cfg.get("load_size")
    if load_size is None:
        return None
    return int(load_size)


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset_config = load_yaml(args.dataset_config)
    print(f"Run name: {train_config['name']}")
    print(f"Direction reference: {train_config['inference_reference']['observed_test_direction']}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoint tag: {args.checkpoint_tag}")
    expected_files = [
        args.checkpoint_dir / f"{args.checkpoint_tag}_net_G_A.pth",
        args.checkpoint_dir / f"{args.checkpoint_tag}_net_G_B.pth",
        args.checkpoint_dir / f"{args.checkpoint_tag}_net_D_A.pth",
        args.checkpoint_dir / f"{args.checkpoint_tag}_net_D_B.pth",
    ]
    for path in expected_files:
        print(f"Expected: {path}")
    if args.dry_run:
        print("Dry run completed. Inference loop is not implemented yet.")
        return
    if args.train_manifest is not None:
        dataset = T1T2PairDataset.from_manifest(args.train_manifest)
    else:
        split_key = "training_root" if args.dataset_split == "training" else "generation_test_root"
        dataset = T1T2PairDataset.from_raw_root(
            Path(dataset_config["data"][split_key]),
            source=dataset_config["modalities"]["source"],
            target=dataset_config["modalities"]["target"],
            resize_to=_resize_to(dataset_config),
        )
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
    manifest_rows: list[dict[str, object]] = []
    limit = min(args.limit, len(dataset))
    print(f"Pairs available: {len(dataset)}")
    print(f"Generating {limit} paired slices into {args.output_dir}")
    for index in range(limit):
        sample = dataset[index]
        if args.direction == "AtoB":
            fake = trainer.infer_AtoB(sample["source"])
            real_image = _slice_to_rgb_uint8(sample["target"])
        else:
            fake = trainer.infer_BtoA(sample["target"])
            real_image = _slice_to_rgb_uint8(sample["source"])
        fake_image = _tensor_to_uint8_image(fake)
        real_path, fake_path = _save_pair(
            args.output_dir,
            sample["patient_id"],
            int(sample["slice_index"]),
            real_kind,
            fake_kind,
            real_image,
            fake_image,
        )
        manifest_rows.append(
            {
                "pair_id": f"{sample['patient_id']}_{sample['slice_index']}",
                "patient_id": sample["patient_id"],
                "slice_index": sample["slice_index"],
                "direction": args.direction,
                "real_kind": real_kind,
                "fake_kind": fake_kind,
                "real_path": str(real_path),
                "fake_path": str(fake_path),
            }
        )
        print(
            f"saved patient={sample['patient_id']} slice={sample['slice_index']} "
            f"real={real_path.name} fake={fake_path.name}"
        )
    manifest_path = args.output_dir / "inference_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
