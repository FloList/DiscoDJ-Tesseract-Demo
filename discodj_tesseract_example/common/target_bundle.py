"""Shared target-bundle schema helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import numpy as np

REQUIRED_TARGET_BUNDLE_KEYS = (
    "target",
    "target_observation",
    "initial_white_noise_fourier",
    "target_base_white_noise_fourier",
    "target_cosmological_base",
    "k_grid",
    "linear_power_grid",
    "rfft_factor",
    "uv_filter",
    "noise_sigma",
    "n2",
    "k_nyquist",
    "target_seed",
    "initial_seed",
    "res",
)

OPTIONAL_MODEL_PARAMETER_KEYS = (
    "model_n_steps",
    "model_boxsize",
    "model_a_ini",
    "model_a_end",
    "model_lpt_order",
    "model_mass_assignment_order",
)


def validate_target_bundle_arrays(arrays: Mapping[str, object]) -> None:
    """Validate required target-bundle keys and core shape consistency."""
    missing = [key for key in REQUIRED_TARGET_BUNDLE_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"target bundle is missing keys: {missing}")

    target_shape = np.asarray(arrays["target"]).shape
    if len(target_shape) != 3 or len(set(target_shape)) != 1:
        raise ValueError(f"target must be a cubic 3D field, got {target_shape}")
    res = target_shape[0]

    if np.asarray(arrays["target_observation"]).shape != target_shape:
        raise ValueError("target_observation shape does not match target")
    if np.asarray(arrays["target_cosmological_base"]).shape != target_shape:
        raise ValueError("target_cosmological_base shape does not match target")

    rfft_shape = (res, res, res // 2 + 1)
    if np.asarray(arrays["k_grid"]).shape != rfft_shape:
        raise ValueError("k_grid shape does not match target resolution")
    if np.asarray(arrays["linear_power_grid"]).shape != rfft_shape:
        raise ValueError("linear_power_grid shape does not match target resolution")
    if np.asarray(arrays["rfft_factor"]).shape != rfft_shape:
        raise ValueError("rfft_factor shape does not match target resolution")
    if np.asarray(arrays["uv_filter"]).shape != rfft_shape:
        raise ValueError("uv_filter shape does not match target resolution")
    if np.asarray(arrays["initial_white_noise_fourier"]).shape != (*rfft_shape, 2):
        raise ValueError("initial_white_noise_fourier shape does not match target resolution")
    if np.asarray(arrays["target_base_white_noise_fourier"]).shape != (*rfft_shape, 2):
        raise ValueError("target_base_white_noise_fourier shape does not match target resolution")

    present_model_keys = [key for key in OPTIONAL_MODEL_PARAMETER_KEYS if key in arrays]
    if present_model_keys and len(present_model_keys) != len(OPTIONAL_MODEL_PARAMETER_KEYS):
        missing_model_keys = [
            key for key in OPTIONAL_MODEL_PARAMETER_KEYS if key not in arrays
        ]
        raise ValueError(f"target bundle is missing model parameter keys: {missing_model_keys}")
    for key in present_model_keys:
        if np.asarray(arrays[key]).shape != ():
            raise ValueError(f"{key} must be a scalar")


def save_target_bundle(path: Path, **arrays: object) -> None:
    """Validate and save a compressed target bundle."""
    validate_target_bundle_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
