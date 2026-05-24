"""Fourier-space helpers shared by the DiscoDJ examples."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from discodj.core.grids import get_fourier_grid
from discodj.core.grids import enforce_hermitian_symmetry


def zero_dc_mode(packed_modes: np.ndarray) -> np.ndarray:
    """Return packed rFFT modes with the DC component set to zero."""
    modes = np.array(packed_modes, copy=True)
    modes[0, 0, 0, :] = 0.0
    return modes.astype(np.float32)


def zero_unsupported_modes(packed_modes: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Return packed rFFT modes with all unsupported Fourier cells set to zero."""
    modes = np.array(packed_modes, copy=True)
    mask = np.asarray(support_mask, dtype=bool)
    if modes.shape[:3] != mask.shape:
        raise ValueError(
            "support_mask shape must match the packed rFFT grid, "
            f"got {mask.shape} for modes {modes.shape}"
        )
    modes[~mask] = 0.0
    return modes.astype(np.float32)


def make_fourier_white_noise(seed: int, res: int) -> np.ndarray:
    """Sample real-space white noise and return packed rFFT modes."""
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(size=(res, res, res)).astype(np.float32)
    modes = np.fft.rfftn(white_noise, norm="ortho").astype(np.complex64)
    return np.stack([modes.real, modes.imag], axis=-1).astype(np.float32)


def rfft_factor_grid(k_grid: jax.Array) -> jax.Array:
    """Return multiplicity factors for an rFFT grid."""
    rfft_factor = jnp.ones_like(k_grid, dtype=jnp.float32)
    return rfft_factor.at[..., 1:-1].set(2.0)


def make_fourier_helpers(
    *,
    res: int,
    boxsize: float,
    noise_sigma: float,
    linear_power_grid: jax.Array,
) -> dict[str, np.ndarray | float]:
    """Build the standard target-bundle Fourier helper arrays."""
    k_grid = get_fourier_grid(
        [res] * 3,
        boxsize=boxsize,
        sparse_k_vecs=True,
        full=False,
        dtype_num=32,
        relative=False,
        with_jax=True,
    )["|k|"]
    rfft_factor = rfft_factor_grid(k_grid)
    k_nyquist = np.pi * res / boxsize
    uv_filter = (k_grid > 0.0) & (k_grid < k_nyquist)
    n2 = noise_sigma**2 / (res / boxsize) ** 3

    return {
        "k_grid": np.asarray(k_grid, dtype=np.float32),
        "linear_power_grid": np.asarray(linear_power_grid, dtype=np.float32),
        "rfft_factor": np.asarray(rfft_factor, dtype=np.float32),
        "uv_filter": np.asarray(uv_filter),
        "k_nyquist": np.asarray(k_nyquist, dtype=np.float32),
        "n2": np.asarray(n2, dtype=np.float32),
        "noise_sigma": np.asarray(noise_sigma, dtype=np.float32),
    }


def helpers_from_pk_table(
    *,
    pk_table: dict[str, jax.Array],
    res: int,
    boxsize: float,
    noise_sigma: float,
) -> dict[str, np.ndarray | float]:
    """Interpolate a DiscoDJ power table onto the standard rFFT helper grid."""
    k_grid = get_fourier_grid(
        [res] * 3,
        boxsize=boxsize,
        sparse_k_vecs=True,
        full=False,
        dtype_num=32,
        relative=False,
        with_jax=True,
    )["|k|"]
    pk_grid = jnp.interp(k_grid, pk_table["k"], pk_table["Pk"])
    return make_fourier_helpers(
        res=res,
        boxsize=boxsize,
        noise_sigma=noise_sigma,
        linear_power_grid=pk_grid,
    )


def linear_delta_from_white_noise_modes(
    white_noise_fourier: jax.Array,
    *,
    linear_power_grid: jax.Array,
    k_grid: jax.Array,
    boxsize: float,
) -> jax.Array:
    """Convert packed Fourier white noise into a linear density field."""
    res = white_noise_fourier.shape[0]
    modes_complex = white_noise_fourier[..., 0] + 1j * white_noise_fourier[..., 1]
    modes_complex = enforce_hermitian_symmetry(modes_complex, with_jax=True)
    norm_fac = (res / float(boxsize)) ** 3
    linear_delta_scale = jnp.sqrt(jnp.asarray(linear_power_grid, dtype=jnp.float32) * norm_fac)
    linear_delta_scale = jnp.where(jnp.asarray(k_grid) > 0.0, linear_delta_scale, 0.0)
    modes_complex = modes_complex / jnp.sqrt(2.0) * res**1.5
    delta_k = modes_complex * linear_delta_scale
    return jnp.fft.irfftn(delta_k, s=(res, res, res), axes=(0, 1, 2)).astype(jnp.float32)
