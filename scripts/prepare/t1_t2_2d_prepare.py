"""
Prepare paired 2D T1/T2 slices from raw `.nii.gz` volumes.

The current implementation targets the first runnable local baseline:

- per-patient T1 / T2 volume pairing
- axial full-slice traversal
- paired `A` / `B` PNG export
- manifest generation for reproducibility
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from imageio import v2 as imageio


@dataclass(frozen=True)
class SliceRecord:
    patient_id: str
    slice_index: int
    source_modality: str
    target_modality: str
    source_volume: str
    target_volume: str
    source_slice_path: str
    target_slice_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired axial T1/T2 slices from raw NIfTI volumes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing per-patient NIfTI files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for derived paired 2D data.",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        help="Logical split name, for example train or generation_test.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Write paired PNG slice files in addition to the manifest.",
    )
    parser.add_argument(
        "--source-dir-name",
        default=None,
        help="Override the source-side output directory name.",
    )
    parser.add_argument(
        "--target-dir-name",
        default=None,
        help="Override the target-side output directory name.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write source/target directories directly under output-root instead "
            "of nesting under split-name. Useful for legacy CycleGAN layouts "
            "such as trainA/trainB/testA/testB."
        ),
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional cap for quick smoke-test preparation.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip slices where both source and target are constant-zero.",
    )
    return parser.parse_args()


def find_patient_dirs(input_root: Path) -> list[Path]:
    patient_dirs = [path for path in input_root.iterdir() if path.is_dir()]
    patient_dirs.sort()
    return patient_dirs


def volume_paths(patient_dir: Path) -> tuple[Path, Path] | None:
    t1 = sorted(patient_dir.glob("*_t1.nii.gz"))
    t2 = sorted(patient_dir.glob("*_t2.nii.gz"))
    if not t1 or not t2:
        return None
    return t1[0], t2[0]


def default_dir_names(split_name: str, flat_output: bool) -> tuple[str, str]:
    if not flat_output:
        return "A", "B"
    normalized = split_name.lower()
    if normalized == "train":
        return "trainA", "trainB"
    if normalized in {"generation_test", "test"}:
        return "testA", "testB"
    return f"{split_name}A", f"{split_name}B"


def output_layout(
    output_root: Path,
    split_name: str,
    source_dir_name: str | None,
    target_dir_name: str | None,
    flat_output: bool,
) -> tuple[Path, Path, Path]:
    default_source, default_target = default_dir_names(split_name, flat_output)
    resolved_source = source_dir_name or default_source
    resolved_target = target_dir_name or default_target
    if flat_output:
        split_root = output_root
        manifest_path = output_root / f"{split_name}_manifest.csv"
    else:
        split_root = output_root / split_name
        manifest_path = split_root / "manifest.csv"
    source_root = split_root / resolved_source
    target_root = split_root / resolved_target
    source_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    return source_root, target_root, manifest_path


def load_volume(path: Path) -> np.ndarray:
    volume = nib.load(str(path)).get_fdata()
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape} for {path}")
    return volume


def legacy_orient(slice_2d: np.ndarray) -> np.ndarray:
    return np.flipud(slice_2d.T)


def slice_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    data = np.asarray(slice_2d, dtype=np.float32)
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        return np.zeros(data.shape, dtype=np.uint8)
    finite = data[finite_mask]
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value <= min_value:
        return np.zeros(data.shape, dtype=np.uint8)
    normalized = (data - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def slice_paths(source_root: Path, target_root: Path, patient_id: str, slice_index: int) -> tuple[Path, Path]:
    filename = f"{patient_id}_img_{slice_index}.png"
    return source_root / filename, target_root / filename


def build_records(
    patient_dirs: Iterable[Path],
    source_root: Path,
    target_root: Path,
    write_png: bool,
    skip_empty: bool,
) -> list[SliceRecord]:
    records: list[SliceRecord] = []
    for patient_dir in patient_dirs:
        paths = volume_paths(patient_dir)
        if paths is None:
            continue
        t1_path, t2_path = paths
        patient_id = patient_dir.name
        t1_volume = load_volume(t1_path)
        t2_volume = load_volume(t2_path)
        if t1_volume.shape != t2_volume.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: {t1_volume.shape} vs {t2_volume.shape}"
            )
        for slice_index in range(t1_volume.shape[2]):
            source_slice = legacy_orient(t1_volume[:, :, slice_index])
            target_slice = legacy_orient(t2_volume[:, :, slice_index])
            if skip_empty and np.all(source_slice == 0) and np.all(target_slice == 0):
                continue
            source_path, target_path = slice_paths(
                source_root, target_root, patient_id, slice_index
            )
            if write_png:
                imageio.imwrite(source_path, slice_to_uint8(source_slice))
                imageio.imwrite(target_path, slice_to_uint8(target_slice))
            records.append(
                SliceRecord(
                    patient_id=patient_id,
                    slice_index=slice_index,
                    source_modality="t1",
                    target_modality="t2",
                    source_volume=str(t1_path),
                    target_volume=str(t2_path),
                    source_slice_path=str(source_path),
                    target_slice_path=str(target_path),
                )
            )
    return records


def write_manifest(manifest_path: Path, records: Iterable[SliceRecord]) -> int:
    fieldnames = [
        "patient_id",
        "slice_index",
        "source_modality",
        "target_modality",
        "source_volume",
        "target_volume",
        "source_slice_path",
        "target_slice_path",
    ]
    count = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    patient_dirs = find_patient_dirs(args.input_root)
    if args.limit_cases is not None:
        patient_dirs = patient_dirs[: args.limit_cases]
    source_root, target_root, manifest_path = output_layout(
        args.output_root,
        args.split_name,
        args.source_dir_name,
        args.target_dir_name,
        args.flat_output,
    )
    count = write_manifest(
        manifest_path,
        build_records(
            patient_dirs,
            source_root,
            target_root,
            write_png=args.write_png,
            skip_empty=args.skip_empty,
        ),
    )
    print(f"Wrote manifest: {manifest_path}")
    print(f"Records: {count}")
    if args.write_png:
        print(f"Wrote PNG slices under: {source_root.parent}")
    else:
        print("Manifest written without PNG export.")


if __name__ == "__main__":
    main()
