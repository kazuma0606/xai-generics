from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairMetrics:
    rmse: float
    mutual_information: float
    psnr: float
    ssim: float


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        gray = (
            0.2989 * image[:, :, 0]
            + 0.5870 * image[:, :, 1]
            + 0.1140 * image[:, :, 2]
        )
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return np.clip(gray, 0, 255).astype(np.uint8)


def rmse(real: np.ndarray, fake: np.ndarray) -> float:
    diff = real.astype(np.float32) - fake.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))


def mutual_information(
    real: np.ndarray,
    fake: np.ndarray,
    bins: int,
    mutual_info_score: Any,
) -> float:
    real_gray = to_grayscale_uint8(real)
    fake_gray = to_grayscale_uint8(fake)
    real_binned = np.floor(real_gray.astype(np.float32) * bins / 256.0).astype(np.int32)
    fake_binned = np.floor(fake_gray.astype(np.float32) * bins / 256.0).astype(np.int32)
    real_binned = np.clip(real_binned, 0, bins - 1)
    fake_binned = np.clip(fake_binned, 0, bins - 1)
    return float(mutual_info_score(real_binned.reshape(-1), fake_binned.reshape(-1)))


def psnr(real: np.ndarray, fake: np.ndarray, peak_signal_noise_ratio: Any) -> float:
    return float(peak_signal_noise_ratio(real, fake, data_range=255.0))


def ssim(real: np.ndarray, fake: np.ndarray, structural_similarity: Any) -> float:
    channel_axis = -1 if real.ndim == 3 else None
    return float(
        structural_similarity(real, fake, data_range=255.0, channel_axis=channel_axis)
    )


def compute_pair_metrics(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    bins: int,
) -> PairMetrics:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from sklearn.metrics import mutual_info_score
    except ImportError as exc:
        raise SystemExit(
            "scikit-image and scikit-learn are required for image metric evaluation."
        ) from exc

    return PairMetrics(
        rmse=rmse(real, fake),
        mutual_information=mutual_information(real, fake, bins, mutual_info_score),
        psnr=psnr(real, fake, peak_signal_noise_ratio),
        ssim=ssim(real, fake, structural_similarity),
    )
