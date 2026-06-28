from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from xai_generics.config import load_yaml
from xai_generics.data.t1_t2_pairs import T1T2PairDataset
from xai_generics.evaluation.image_metrics import compute_pair_metrics
from xai_generics.models.cyclegan import CycleGANBundle
from xai_generics.training.cyclegan_trainer import CycleGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local T1/T2 training entrypoint scaffold."
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
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional manifest.csv produced by scripts/prepare.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset access without starting a training loop.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on total training steps across all epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed used for the dataset order.",
    )
    parser.add_argument(
        "--checkpoint-tag",
        default="latest",
        help="Checkpoint prefix used for the final saved weights.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint/state matching --resume-tag if present.",
    )
    parser.add_argument(
        "--resume-tag",
        default="latest",
        help="Checkpoint tag to resume from when --resume is enabled.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override the configured run root.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Override the configured checkpoint root.",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=None,
        help="Override the configured sample root.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Override the configured log root.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of slices used for training.",
    )
    parser.add_argument(
        "--validate-every-epoch",
        action="store_true",
        help="Run full generation-side validation after each epoch.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Optional cap on validation slices.",
    )
    parser.add_argument(
        "--validation-output-root",
        type=Path,
        default=None,
        help="Directory for validation CSV/plot outputs.",
    )
    return parser.parse_args()


def _slice_to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got {image.shape}")
    image = np.clip(image, 0.0, 255.0)
    image = np.rint(image).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2)


def _tensor_to_uint8_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor output, got {image.shape}")
    image = np.clip((image + 1.0) * 127.5, 0.0, 255.0)
    image = np.rint(image).astype(np.uint8)
    return np.transpose(image, (1, 2, 0))


def _write_sample(output_dir: Path, sample: dict[str, object], fake: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = str(sample["patient_id"])
    slice_index = int(sample["slice_index"])
    real = _slice_to_rgb_uint8(np.asarray(sample["target"], dtype=np.float32))
    real_path = output_dir / f"{patient_id}_{slice_index}_real_B.png"
    fake_path = output_dir / f"{patient_id}_{slice_index}_fake_B.png"
    imageio.imwrite(real_path, real)
    imageio.imwrite(fake_path, _tensor_to_uint8_image(fake))


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


def _load_validation_dataset(dataset_config: dict[str, object]) -> T1T2PairDataset:
    resize_to = _resize_to(dataset_config)
    return T1T2PairDataset.from_raw_root(
        Path(dataset_config["data"]["generation_test_root"]),
        source=dataset_config["modalities"]["source"],
        target=dataset_config["modalities"]["target"],
        resize_to=resize_to,
    )


def _append_log(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _append_validation_log(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _evaluate_validation(
    trainer: CycleGANTrainer,
    dataset: T1T2PairDataset,
    *,
    epoch: int,
    step: int,
    limit: int | None,
) -> dict[str, float]:
    total_rmse = 0.0
    total_mi = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    max_items = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(max_items):
        sample = dataset[index]
        fake = trainer.infer_AtoB(sample["source"])
        fake_image = _tensor_to_uint8_image(fake)
        real_image = np.asarray(sample["target"], dtype=np.float32)
        if real_image.ndim == 2:
            real_image = np.repeat(np.rint(np.clip(real_image, 0.0, 255.0)).astype(np.uint8)[:, :, None], 3, axis=2)
        metrics = compute_pair_metrics(real_image, fake_image, bins=64)
        total_rmse += metrics.rmse
        total_mi += metrics.mutual_information
        total_psnr += metrics.psnr
        total_ssim += metrics.ssim
        count += 1
    if count == 0:
        raise SystemExit("Validation dataset is empty.")
    return {
        "epoch": float(epoch),
        "step": float(step),
        "pair_count": float(count),
        "rmse_mean": total_rmse / count,
        "mutual_information_mean": total_mi / count,
        "psnr_mean": total_psnr / count,
        "ssim_mean": total_ssim / count,
    }


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset_config = load_yaml(args.dataset_config)
    if args.train_manifest is not None:
        dataset = T1T2PairDataset.from_manifest(args.train_manifest)
        dataset_source = str(args.train_manifest)
    else:
        resize_to = _resize_to(dataset_config)
        dataset = T1T2PairDataset.from_raw_root(
            Path(dataset_config["data"]["training_root"]),
            source=dataset_config["modalities"]["source"],
            target=dataset_config["modalities"]["target"],
            resize_to=resize_to,
        )
        dataset_source = dataset_config["data"]["training_root"]

    print(f"Run name: {train_config['name']}")
    print(f"Model type: {train_config['model']['type']}")
    print(f"Direction: {train_config['training']['direction']}")
    print(f"Dataset source: {dataset_source}")
    print(f"Pairs available: {len(dataset)}")
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty.")
    sample = dataset[0]
    print(
        "First sample: "
        f"patient={sample['patient_id']} slice={sample['slice_index']} "
        f"source_shape={sample['source'].shape} target_shape={sample['target'].shape}"
    )
    try:
        bundle = CycleGANBundle.from_config(train_config)
        counts = bundle.parameter_counts()
        print(f"Runtime device: {bundle.device}")
        print(
            "Parameter counts: "
            f"G_A={counts['netG_A']} G_B={counts['netG_B']} "
            f"D_A={counts['netD_A']} D_B={counts['netD_B']}"
        )
    except SystemExit as exc:
        print(str(exc))
    if args.dry_run:
        print("Dry run completed. Training loop is not implemented yet.")
        return
    trainer = CycleGANTrainer.from_config(train_config)
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = int(train_config["training"]["n_epochs"]) + int(
            train_config["training"]["n_epochs_decay"]
        )
    logging_cfg = train_config["logging"]
    run_root = args.run_root or Path(train_config["artifacts"]["run_root"])
    checkpoint_root = args.checkpoint_root or Path(train_config["artifacts"]["checkpoint_root"])
    sample_root = args.sample_root or Path(train_config["artifacts"]["sample_root"])
    log_root = args.log_root or Path(train_config["artifacts"]["log_root"])
    validation_root = args.validation_output_root or (log_root / "validation")
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    sample_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)
    history_path = log_root / "training_log.csv"
    validation_history_path = validation_root / "validation_log.csv"

    start_epoch = 1
    global_step = 0
    if args.resume:
        resume_weights = checkpoint_root / f"{args.resume_tag}_net_G_A.pth"
        resume_state = checkpoint_root / f"{args.resume_tag}_state.pt"
        if resume_weights.exists():
            trainer.bundle.load_checkpoint(checkpoint_root, args.resume_tag, strict=True)
            print(f"Resumed model weights from tag={args.resume_tag}")
        else:
            print(f"No checkpoint weights found for tag={args.resume_tag}; starting fresh.")
        if resume_state.exists():
            state = trainer.bundle.load_training_state(checkpoint_root, args.resume_tag)
            start_epoch = int(state.get("epoch", 0)) + 1
            global_step = int(state.get("global_step", 0))
            print(
                f"Resumed optimizer state from tag={args.resume_tag} "
                f"at epoch={start_epoch} global_step={global_step}"
            )
        else:
            print(
                f"No training state found for tag={args.resume_tag}; "
                "optimizer state will start fresh."
            )

    indices = list(range(len(dataset)))
    if args.max_samples is not None:
        indices = indices[: max(0, args.max_samples)]
        if not indices:
            raise SystemExit("--max-samples produced an empty training subset.")
    rng = random.Random(args.seed)
    history: list[dict[str, object]] = []
    validation_dataset = _load_validation_dataset(dataset_config) if args.validate_every_epoch else None
    for epoch in range(start_epoch, epochs + 1):
        rng.shuffle(indices)
        epoch_losses: list[dict[str, float]] = []
        for position, index in enumerate(indices):
            sample = dataset[index]
            metrics = trainer.train_step(sample["source"], sample["target"])
            epoch_losses.append(metrics)
            global_step += 1
            history.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "patient_id": sample["patient_id"],
                    "slice_index": sample["slice_index"],
                    "loss_G": metrics["loss_G"],
                    "loss_D": metrics["loss_D"],
                    "loss_G_A2B": metrics["loss_G_A2B"],
                    "loss_G_B2A": metrics["loss_G_B2A"],
                    "loss_D_A2B": metrics["loss_D_A2B"],
                    "loss_D_B2A": metrics["loss_D_B2A"],
                    "loss_cycle_A": metrics["loss_cycle_A"],
                    "loss_cycle_B": metrics["loss_cycle_B"],
                    "loss_idt_A": metrics["loss_idt_A"],
                    "loss_idt_B": metrics["loss_idt_B"],
                }
            )
            if global_step % int(logging_cfg["print_freq"]) == 0:
                print(
                    f"epoch={epoch} step={global_step} patient={sample['patient_id']} "
                    f"slice={sample['slice_index']} loss_G={metrics['loss_G']:.4f} "
                    f"loss_D={metrics['loss_D']:.4f}"
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if epoch_losses:
            mean_loss_g = sum(item["loss_G"] for item in epoch_losses) / len(epoch_losses)
            mean_loss_d = sum(item["loss_D"] for item in epoch_losses) / len(epoch_losses)
            print(
                f"epoch_complete={epoch} steps={len(epoch_losses)} "
                f"mean_loss_G={mean_loss_g:.4f} mean_loss_D={mean_loss_d:.4f}"
            )
        trainer.save(checkpoint_root, f"epoch_{epoch:03d}")
        trainer.save(checkpoint_root, "latest")
        trainer.bundle.save_training_state(
            checkpoint_root,
            f"epoch_{epoch:03d}",
            epoch=epoch,
            global_step=global_step,
            history_path=str(history_path),
        )
        trainer.bundle.save_training_state(
            checkpoint_root,
            "latest",
            epoch=epoch,
            global_step=global_step,
            history_path=str(history_path),
        )
        sample_item = dataset[indices[0]]
        fake = trainer.infer_AtoB(sample_item["source"])
        _write_sample(sample_root / f"epoch_{epoch:03d}", sample_item, fake)
        _append_log(history_path, history)
        history.clear()
        if args.validate_every_epoch and validation_dataset is not None:
            validation_summary = _evaluate_validation(
                trainer,
                validation_dataset,
                epoch=epoch,
                step=global_step,
                limit=args.validation_limit,
            )
            _append_validation_log(validation_history_path, [validation_summary])
            print(
                f"validation epoch={epoch} step={global_step} "
                f"rmse={validation_summary['rmse_mean']:.4f} "
                f"mi={validation_summary['mutual_information_mean']:.4f} "
                f"psnr={validation_summary['psnr_mean']:.4f} "
                f"ssim={validation_summary['ssim_mean']:.4f}"
            )
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    print(f"Training complete. Checkpoints: {checkpoint_root}")
    print(f"Samples: {sample_root}")
    print(f"Logs: {history_path}")
    if args.validate_every_epoch:
        print(f"Validation logs: {validation_history_path}")


if __name__ == "__main__":
    main()
