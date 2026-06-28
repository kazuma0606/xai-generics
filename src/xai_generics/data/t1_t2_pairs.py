from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nibabel as nib
import numpy as np
from skimage.transform import resize


@dataclass(frozen=True)
class PairRecord:
    patient_id: str
    slice_index: int
    source_modality: str
    target_modality: str
    source_volume: str
    target_volume: str
    source_slice_path: str | None = None
    target_slice_path: str | None = None


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


def read_manifest(manifest_path: Path) -> list[PairRecord]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                PairRecord(
                    patient_id=row["patient_id"],
                    slice_index=int(row["slice_index"]),
                    source_modality=row["source_modality"],
                    target_modality=row["target_modality"],
                    source_volume=row["source_volume"],
                    target_volume=row["target_volume"],
                    source_slice_path=row.get("source_slice_path") or None,
                    target_slice_path=row.get("target_slice_path") or None,
                )
            )
    return rows


def _patient_dirs(root: Path) -> list[Path]:
    patient_dirs = [path for path in root.iterdir() if path.is_dir()]
    patient_dirs.sort()
    return patient_dirs


def _volume_paths(patient_dir: Path, source: str, target: str) -> tuple[Path, Path] | None:
    source_paths = sorted(patient_dir.glob(f"*_{source}.nii.gz"))
    target_paths = sorted(patient_dir.glob(f"*_{target}.nii.gz"))
    if not source_paths or not target_paths:
        return None
    return source_paths[0], target_paths[0]


def build_records_from_raw(
    input_root: Path,
    source: str = "t1",
    target: str = "t2",
) -> list[PairRecord]:
    rows: list[PairRecord] = []
    for patient_dir in _patient_dirs(input_root):
        volume_paths = _volume_paths(patient_dir, source, target)
        if volume_paths is None:
            continue
        source_volume, target_volume = volume_paths
        source_data = nib.load(str(source_volume)).get_fdata()
        target_data = nib.load(str(target_volume)).get_fdata()
        if source_data.ndim != 3 or target_data.ndim != 3:
            raise ValueError(
                f"Expected 3D volumes for {patient_dir.name}, got "
                f"{source_data.shape} and {target_data.shape}"
            )
        if source_data.shape != target_data.shape:
            raise ValueError(
                f"Shape mismatch for {patient_dir.name}: "
                f"{source_data.shape} vs {target_data.shape}"
            )
        for slice_index in range(source_data.shape[2]):
            rows.append(
                PairRecord(
                    patient_id=patient_dir.name,
                    slice_index=slice_index,
                    source_modality=source,
                    target_modality=target,
                    source_volume=str(source_volume),
                    target_volume=str(target_volume),
                )
            )
    return rows


def load_pair_numpy(record: PairRecord) -> tuple[np.ndarray, np.ndarray]:
    if record.source_slice_path and record.target_slice_path:
        source = np.asarray(imageio.imread(record.source_slice_path), dtype=np.float32)
        target = np.asarray(imageio.imread(record.target_slice_path), dtype=np.float32)
        return source, target

    source_volume = nib.load(record.source_volume).get_fdata()
    target_volume = nib.load(record.target_volume).get_fdata()
    source_slice = legacy_orient(source_volume[:, :, record.slice_index])
    target_slice = legacy_orient(target_volume[:, :, record.slice_index])
    return slice_to_uint8(source_slice).astype(np.float32), slice_to_uint8(
        target_slice
    ).astype(np.float32)


class T1T2PairDataset:
    def __init__(self, records: list[PairRecord], resize_to: int | None = None) -> None:
        self.records = records
        self.resize_to = resize_to

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "T1T2PairDataset":
        return cls(read_manifest(manifest_path))

    @classmethod
    def from_raw_root(
        cls,
        input_root: Path,
        source: str = "t1",
        target: str = "t2",
        resize_to: int | None = None,
    ) -> "T1T2PairDataset":
        return cls(
            build_records_from_raw(input_root, source=source, target=target),
            resize_to=resize_to,
        )

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if self.resize_to is None:
            return image
        if image.shape == (self.resize_to, self.resize_to):
            return image
        resized = resize(
            image,
            (self.resize_to, self.resize_to),
            preserve_range=True,
            anti_aliasing=True,
        )
        return np.asarray(resized, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source, target = load_pair_numpy(record)
        source = self._resize(source)
        target = self._resize(target)
        return {
            "patient_id": record.patient_id,
            "slice_index": record.slice_index,
            "source_modality": record.source_modality,
            "target_modality": record.target_modality,
            "source": source,
            "target": target,
        }
