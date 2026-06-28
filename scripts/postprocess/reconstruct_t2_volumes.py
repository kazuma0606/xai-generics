"""
Reconstruct case-level real and generated T2 volumes from per-slice PNG files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nrrd
from imageio import v2 as imageio


SLICE_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<slice_index>\d+)_(?P<kind>real_B|fake_B)$")


@dataclass(frozen=True)
class SliceFile:
    prefix: str
    slice_index: int
    kind: str
    path: Path


def direction_suffixes(direction: str) -> tuple[str, str]:
    if direction == "AtoB":
        return "real_B", "fake_B"
    if direction == "BtoA":
        return "real_A", "fake_A"
    raise ValueError(f"Unsupported direction: {direction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct real_T2.nrrd and fake_T2_epoch_200.nrrd from PNG slices."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory containing per-slice image files such as *_1_real_B.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write reconstructed NRRD files.",
    )
    parser.add_argument(
        "--direction",
        choices=["AtoB", "BtoA"],
        default="AtoB",
        help="CycleGAN generation direction used to choose which real/fake slice pair to reconstruct.",
    )
    parser.add_argument(
        "--real-kind",
        default=None,
        help="Optional override for the reference slice kind, for example real_B or real_A.",
    )
    parser.add_argument(
        "--fake-kind",
        default=None,
        help="Optional override for the generated slice kind, for example fake_B or fake_A.",
    )
    parser.add_argument(
        "--real-output-name",
        default="real_T2.nrrd",
        help="Output filename for the reconstructed real volume.",
    )
    parser.add_argument(
        "--fake-output-name",
        default="fake_T2_epoch_200.nrrd",
        help="Output filename for the reconstructed generated volume.",
    )
    return parser.parse_args()


def parse_slice_file(path: Path) -> SliceFile | None:
    match = SLICE_PATTERN.match(path.stem)
    if match is None:
        return None
    return SliceFile(
        prefix=match.group("prefix"),
        slice_index=int(match.group("slice_index")),
        kind=match.group("kind"),
        path=path,
    )


def grayscale_slice(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
        if np.array_equal(red, green) and np.array_equal(green, blue):
            gray = red
        else:
            gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return np.asarray(gray, dtype=np.float32)


def collect_slices(
    image_root: Path,
    real_kind: str,
    fake_kind: str,
) -> tuple[list[SliceFile], list[SliceFile]]:
    real_slices: list[SliceFile] = []
    fake_slices: list[SliceFile] = []
    for path in sorted(image_root.glob("*.png")):
        parsed = parse_slice_file(path)
        if parsed is None:
            continue
        if parsed.kind == real_kind:
            real_slices.append(parsed)
        elif parsed.kind == fake_kind:
            fake_slices.append(parsed)
    real_slices.sort(key=lambda item: item.slice_index)
    fake_slices.sort(key=lambda item: item.slice_index)
    return real_slices, fake_slices


def validate_slice_sets(real_slices: list[SliceFile], fake_slices: list[SliceFile]) -> None:
    if not real_slices:
        raise ValueError("No real_B slices found.")
    if not fake_slices:
        raise ValueError("No fake_B slices found.")
    real_indices = [item.slice_index for item in real_slices]
    fake_indices = [item.slice_index for item in fake_slices]
    if real_indices != fake_indices:
        raise ValueError(
            f"Slice index mismatch between real and fake sets: {real_indices[:5]} vs {fake_indices[:5]}"
        )


def stack_volume(slices: list[SliceFile]) -> np.ndarray:
    stack: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for item in slices:
        image = np.asarray(imageio.imread(item.path))
        gray = grayscale_slice(image)
        if expected_shape is None:
            expected_shape = gray.shape
        elif gray.shape != expected_shape:
            raise ValueError(
                f"Slice shape mismatch for {item.path.name}: {gray.shape} vs {expected_shape}"
            )
        stack.append(gray)
    return np.stack(stack, axis=0)


def write_nrrd(path: Path, volume: np.ndarray) -> None:
    header = {
        "dimension": 3,
        "sizes": list(volume.shape),
        "type": "float",
    }
    nrrd.write(str(path), volume.astype(np.float32), header=header)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_real_kind, default_fake_kind = direction_suffixes(args.direction)
    real_kind = args.real_kind or default_real_kind
    fake_kind = args.fake_kind or default_fake_kind
    real_slices, fake_slices = collect_slices(args.image_root, real_kind, fake_kind)
    validate_slice_sets(real_slices, fake_slices)
    real_volume = stack_volume(real_slices)
    fake_volume = stack_volume(fake_slices)
    real_output = args.output_dir / args.real_output_name
    fake_output = args.output_dir / args.fake_output_name
    write_nrrd(real_output, real_volume)
    write_nrrd(fake_output, fake_volume)
    print(f"Real slices: {len(real_slices)}")
    print(f"Fake slices: {len(fake_slices)}")
    print(f"Direction: {args.direction}")
    print(f"Reference kind: {real_kind}")
    print(f"Generated kind: {fake_kind}")
    print(f"Volume shape: {real_volume.shape}")
    print(f"Wrote real volume: {real_output}")
    print(f"Wrote fake volume: {fake_output}")


if __name__ == "__main__":
    main()
