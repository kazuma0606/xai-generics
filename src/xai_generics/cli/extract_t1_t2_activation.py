from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from xai_generics.config import load_yaml
from xai_generics.data.t1_t2_pairs import T1T2PairDataset
from xai_generics.models.cyclegan import CycleGANBundle, resolve_submodule
from xai_generics.training.cyclegan_trainer import CycleGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an intermediate activation tensor from a loaded CycleGAN checkpoint."
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/2d/train_t1_t2_cyclegan_fast.yaml"),
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/2d/dataset_t1_t2_fast.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/runs/t1_t2_cyclegan_2d_fast_10s/checkpoints"),
    )
    parser.add_argument(
        "--checkpoint-tag",
        default="latest",
    )
    parser.add_argument(
        "--side",
        choices=["A", "B"],
        default="A",
        help="Generator side to inspect. A means netG_A, B means netG_B.",
    )
    parser.add_argument(
        "--module-path",
        default="6",
        help="Dotted path under the selected generator, for example 4, 6, 6.block.1, or 8.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset index used to drive the forward pass.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["training", "generation_test"],
        default="generation_test",
        help="Which raw root to read when no manifest is provided.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/activations/t1_t2_cyclegan"),
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Print available generator module paths and exit.",
    )
    return parser.parse_args()


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


def _load_dataset(dataset_config: dict[str, object], args: argparse.Namespace) -> T1T2PairDataset:
    if args.train_manifest is not None:
        return T1T2PairDataset.from_manifest(args.train_manifest)
    split_key = "training_root" if args.dataset_split == "training" else "generation_test_root"
    data_cfg = dataset_config["data"]
    modalities = dataset_config["modalities"]
    if not isinstance(data_cfg, dict) or not isinstance(modalities, dict):
        raise ValueError("Invalid dataset config structure.")
    return T1T2PairDataset.from_raw_root(
        Path(data_cfg[split_key]),
        source=str(modalities["source"]),
        target=str(modalities["target"]),
        resize_to=_resize_to(dataset_config),
    )


def _to_uint8_preview(tensor: np.ndarray) -> np.ndarray:
    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array.mean(axis=0)
    elif array.ndim != 2:
        raise ValueError(f"Unsupported activation shape for preview: {array.shape}")
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    values = array[finite]
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (array - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _tensor_summary(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset_config = load_yaml(args.dataset_config)
    dataset = _load_dataset(dataset_config, args)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty.")
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise SystemExit(f"sample_index must be within [0, {len(dataset) - 1}]")

    trainer = CycleGANTrainer.from_config(train_config)
    trainer.bundle.load_checkpoint(args.checkpoint_dir, args.checkpoint_tag, strict=True)
    bundle = trainer.bundle
    root = bundle.netG_A if args.side == "A" else bundle.netG_B
    module_names = bundle.available_module_names(args.side)
    if args.list_modules:
        print(json.dumps(module_names, indent=2))
        return
    target_module = resolve_submodule(root, args.module_path)
    sample = dataset[args.sample_index]
    activation_holder: dict[str, np.ndarray] = {}

    def hook(_module, _inputs, output):  # type: ignore[no-untyped-def]
        activation_holder["value"] = output.detach().cpu().numpy()

    handle = target_module.register_forward_hook(hook)
    try:
        if args.side == "A":
            output = trainer.infer_AtoB(sample["source"])
        else:
            output = trainer.infer_BtoA(sample["target"])
    finally:
        handle.remove()

    activation = activation_holder.get("value")
    if activation is None:
        raise SystemExit(f"No activation captured for module path {args.module_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{sample['patient_id']}_{sample['slice_index']}_{args.side}_{args.module_path.replace('.', '_')}"
    npy_path = args.output_dir / f"{prefix}.npy"
    preview_path = args.output_dir / f"{prefix}_preview.png"
    channel0_path = args.output_dir / f"{prefix}_ch0.png"
    np.save(npy_path, activation)
    preview = _to_uint8_preview(activation)
    imageio.imwrite(preview_path, preview)
    if activation.ndim == 4 and activation.shape[0] == 1:
        channel0 = activation[0, 0]
    elif activation.ndim == 3:
        channel0 = activation[0]
    else:
        channel0 = preview
    imageio.imwrite(channel0_path, _to_uint8_preview(channel0))
    summary_path = args.output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(_tensor_summary(activation), indent=2), encoding="utf-8")
    print(f"Captured module: {args.side}:{args.module_path}")
    print(f"Activation shape: {activation.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Wrote tensor: {npy_path}")
    print(f"Wrote preview: {preview_path}")
    print(f"Wrote channel0 preview: {channel0_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
