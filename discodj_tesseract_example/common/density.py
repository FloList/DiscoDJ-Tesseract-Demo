"""Density-field helpers shared by target generators and notebooks."""
from __future__ import annotations
import numpy as np

DELTA_FLOOR = -1.0
RHO_FLOOR = 0.0


def rms(field: np.ndarray) -> float:
    """Return the RMS after subtracting the mean."""
    centered = field.astype(np.float32) - np.mean(field, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))


def density_from_delta(delta: np.ndarray) -> np.ndarray:
    """Convert density contrast to non-negative density."""
    return np.maximum(1.0 + delta.astype(np.float32), RHO_FLOOR)


def delta_from_density(rho: np.ndarray) -> np.ndarray:
    """Convert density to mean-normalized density contrast."""
    mean_rho = float(np.mean(rho, dtype=np.float64))
    if not np.isfinite(mean_rho) or mean_rho <= 0.0:
        raise ValueError("density has non-positive/non-finite mean")
    delta = rho.astype(np.float32) / np.float32(mean_rho) - np.float32(1.0)
    return np.maximum(delta, DELTA_FLOOR).astype(np.float32)


def add_noise_with_floor(
    target: np.ndarray,
    *,
    seed: int,
    noise_sigma: float,
) -> np.ndarray:
    """Add Gaussian density noise while preserving non-negative density."""
    rng = np.random.default_rng(seed)
    noisy_rho = density_from_delta(target) + rng.normal(
        scale=noise_sigma,
        size=target.shape,
    ).astype(np.float32)
    return delta_from_density(np.maximum(noisy_rho, RHO_FLOOR))


def log_density_for_plot(delta: np.ndarray) -> np.ndarray:
    """Return a stable log-density image for visualization."""
    return np.log10(np.maximum(1.1 + delta, 0.1))
