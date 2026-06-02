#!/usr/bin/env python3
"""Generate the FGPA log_A demo target bundle."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discodj_tesseract_example.common.cosmology import DEFAULT_COSMOLOGY, DEFAULT_MODEL
from discodj_tesseract_example.common.density import log_density_for_plot
from discodj import DiscoDJ
from discodj_tesseract_example.common.fourier import (
    linear_delta_from_white_noise_modes,
    helpers_from_pk_table,
    make_fourier_white_noise,
    zero_unsupported_modes,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["savefig.format"] = "pdf"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TARGET_BUNDLE_HASH_KEY = "target_bundle_content_sha256"


def target_bundle_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(k for k in arrays if k != TARGET_BUNDLE_HASH_KEY):
        arr = np.asarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(arr).view(np.uint8).tobytes())
    return digest.hexdigest()


def load_forward_builder():
    path = HERE / "discodj-forward-external-ics-tesseract" / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location("fgpa_forward_tesseract_api", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_forward


def make_random_skewer_xy(seed: int, res: int, n_skewers: int) -> np.ndarray:
    if n_skewers > res * res:
        raise ValueError("n_skewers cannot exceed res**2")
    rng = np.random.default_rng(seed)
    flat = rng.choice(res * res, size=n_skewers, replace=False)
    return np.stack([flat // res, flat % res], axis=-1).astype(np.int32)


def fgpa_flux_skewers_numpy(
    delta_field: np.ndarray,
    log_a: float,
    beta: float,
    skewer_xy: np.ndarray,
    *,
    rho_floor: float,
) -> np.ndarray:
    delta = np.asarray(delta_field, dtype=np.float32)
    coords = np.asarray(skewer_xy, dtype=np.int32)
    rho = np.maximum(np.float32(1.0) + delta[coords[:, 0], coords[:, 1], :], np.float32(rho_floor))
    tau = np.exp(np.float32(log_a)) * rho ** np.float32(beta)
    return np.exp(-tau).astype(np.float32)


def add_flux_noise(clean_flux: np.ndarray, *, seed: int, noise_sigma: float, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=float(noise_sigma), size=clean_flux.shape)
    return (np.asarray(clean_flux, dtype=dtype) + noise.astype(dtype)).astype(dtype)


def save_preview(path: Path, *, delta_z2: np.ndarray, clean_flux: np.ndarray, noisy_flux: np.ndarray, skewer_xy: np.ndarray) -> None:
    mid = delta_z2.shape[2] // 2
    draw_count = min(6, noisy_flux.shape[0])
    draw_idx = np.linspace(0, noisy_flux.shape[0] - 1, draw_count, dtype=int)
    los = np.arange(noisy_flux.shape[1])

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_density = fig.add_subplot(gs[:, 0])
    ax_image = fig.add_subplot(gs[0, 1:])
    ax_lines = fig.add_subplot(gs[1, 1:])

    image = ax_density.imshow(log_density_for_plot(delta_z2[:, :, mid]).T, origin="lower", cmap="magma")
    ax_density.scatter(skewer_xy[:, 0], skewer_xy[:, 1], s=14, c="cyan", edgecolors="black", linewidths=0.3)
    ax_density.set_title("z=2 density slice with skewer positions")
    ax_density.set_xticks([])
    ax_density.set_yticks([])
    fig.colorbar(image, ax=ax_density, shrink=0.75)

    flux_image = ax_image.imshow(noisy_flux, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_image.set_title("Noisy FGPA flux skewers")
    ax_image.set_xlabel("line-of-sight grid index")
    ax_image.set_ylabel("skewer index")
    fig.colorbar(flux_image, ax=ax_image, shrink=0.85, label="F")

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, draw_count))
    for color, idx in zip(colors, draw_idx, strict=True):
        ax_lines.plot(los, clean_flux[idx], color=color, lw=1.5, label=f"skewer {idx} clean")
        ax_lines.scatter(los, noisy_flux[idx], color=color, s=9, alpha=0.45)
    ax_lines.set_ylim(-0.1, 1.1)
    ax_lines.set_xlabel("line-of-sight grid index")
    ax_lines.set_ylabel("F")
    ax_lines.set_title("Representative flux skewers")
    ax_lines.legend(ncols=2, fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=123)
    parser.add_argument("--initial-seed", type=int, default=17)
    parser.add_argument("--skewer-seed", type=int, default=99)
    parser.add_argument("--noise-seed", type=int, default=100123)
    parser.add_argument("--n-skewers", type=int, default=256)
    parser.add_argument("--flux-noise-sigma", type=float, default=0.03)
    parser.add_argument("--log-a", type=float, default=float(np.log(0.35)))
    parser.add_argument("--beta", type=float, default=1.6)
    parser.add_argument("--log-a-init", type=float, default=float(np.log(0.8)))
    parser.add_argument("--rho-floor", type=float, default=1.0e-3)
    parser.add_argument("--target-tag", type=str, required=True)
    parser.add_argument("--target-root", type=Path, default=HERE / "targets")
    args = parser.parse_args()
    target_tag_path = Path(args.target_tag)
    if target_tag_path.is_absolute() or ".." in target_tag_path.parts:
        raise ValueError("--target-tag must be a relative folder name")
    target_dir = args.target_root / target_tag_path
    args.target_bundle_output = target_dir / "target_bundle.npz"
    args.preview_output = target_dir / "fgpa_loga_target.pdf"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.res <= 0:
        raise ValueError("--res must be positive")
    if not 0 < args.n_skewers <= args.res * args.res:
        raise ValueError("--n-skewers must be between 1 and res**2")
    if args.flux_noise_sigma <= 0.0:
        raise ValueError("--flux-noise-sigma must be positive")
    if args.rho_floor <= 0.0:
        raise ValueError("--rho-floor must be positive")
    if args.beta <= 0.0:
        raise ValueError("--beta must be positive")
    if args.target_seed == args.initial_seed:
        raise ValueError("target seed and initial seed must differ")


def main() -> None:
    args = parse_args()
    validate_args(args)
    field_dtype = np.float32
    jax_field_dtype = jnp.float32
    a_end = 1.0 / 3.0
    dj = DiscoDJ(
        dim=3,
        res=args.res,
        name="fgpa-loga-target",
        device=jax.default_backend(),
        precision="single",
        boxsize=DEFAULT_MODEL.boxsize,
        cosmo=DEFAULT_COSMOLOGY,
    )
    dj = dj.with_timetables(timetable_settings={"a_max": max(1.0, DEFAULT_MODEL.a_end)})
    pk_state = dj.with_linear_ps(transfer_function="Eisenstein-Hu")
    helpers = helpers_from_pk_table(
        pk_table=pk_state._pk_table,
        res=args.res,
        boxsize=DEFAULT_MODEL.boxsize,
        noise_sigma=args.flux_noise_sigma,
    )

    support_mask = np.asarray(helpers["uv_filter"], dtype=bool)
    target_white_noise = zero_unsupported_modes(make_fourier_white_noise(args.target_seed, args.res), support_mask).astype(field_dtype)
    initial_white_noise = zero_unsupported_modes(make_fourier_white_noise(args.initial_seed, args.res), support_mask).astype(field_dtype)
    target_delta_ini = linear_delta_from_white_noise_modes(
        jnp.asarray(target_white_noise, dtype=jax_field_dtype),
        linear_power_grid=jnp.asarray(helpers["linear_power_grid"], dtype=jax_field_dtype),
        k_grid=jnp.asarray(helpers["k_grid"], dtype=jax_field_dtype),
        boxsize=DEFAULT_MODEL.boxsize,
    )
    initial_delta_ini = linear_delta_from_white_noise_modes(
        jnp.asarray(initial_white_noise, dtype=jax_field_dtype),
        linear_power_grid=jnp.asarray(helpers["linear_power_grid"], dtype=jax_field_dtype),
        k_grid=jnp.asarray(helpers["k_grid"], dtype=jax_field_dtype),
        boxsize=DEFAULT_MODEL.boxsize,
    )

    forward = load_forward_builder()(
        args.res,
        DEFAULT_MODEL.n_steps,
        DEFAULT_MODEL.boxsize,
        DEFAULT_MODEL.a_ini,
        a_end,
        1,
        3,
        "lpt",
        "single",
    )
    omega_m = float(DEFAULT_COSMOLOGY["Omega_c"] + DEFAULT_COSMOLOGY["Omega_b"])
    omega_b = float(DEFAULT_COSMOLOGY["Omega_b"])
    sigma8 = float(DEFAULT_COSMOLOGY["sigma8"])
    delta_z2 = np.asarray(forward(jnp.asarray(target_delta_ini, dtype=jax_field_dtype), jnp.asarray(omega_m, dtype=jnp.float32)), dtype=field_dtype)
    initial_delta_z2 = np.asarray(forward(jnp.asarray(initial_delta_ini, dtype=jax_field_dtype), jnp.asarray(omega_m, dtype=jnp.float32)), dtype=field_dtype)

    skewer_xy = make_random_skewer_xy(args.skewer_seed, args.res, args.n_skewers)
    log_a_start = float(np.log(0.25))
    beta_start = 1.2
    clean_flux = fgpa_flux_skewers_numpy(delta_z2, args.log_a, args.beta, skewer_xy, rho_floor=args.rho_floor).astype(field_dtype)
    noisy_flux = add_flux_noise(clean_flux, seed=args.noise_seed, noise_sigma=args.flux_noise_sigma, dtype=field_dtype)
    initial_flux = fgpa_flux_skewers_numpy(initial_delta_z2, log_a_start, beta_start, skewer_xy, rho_floor=args.rho_floor).astype(field_dtype)

    target_bundle_arrays = dict(
        target=clean_flux,
        target_observation=noisy_flux,
        target_flux_clean=clean_flux,
        target_flux_noisy=noisy_flux,
        initial_flux=initial_flux,
        target_delta_z=delta_z2,
        target_delta_z2=delta_z2,
        initial_delta_z=initial_delta_z2,
        initial_delta_z2=initial_delta_z2,
        target_delta_ini=np.asarray(target_delta_ini, dtype=field_dtype),
        initial_delta_ini=np.asarray(initial_delta_ini, dtype=field_dtype),
        initial_white_noise_fourier=initial_white_noise,
        target_base_white_noise_fourier=target_white_noise,
        target_cosmological_base=delta_z2,
        skewer_xy=skewer_xy,
        skewer_x=skewer_xy[:, 0].astype(np.int32),
        skewer_y=skewer_xy[:, 1].astype(np.int32),
        k_grid=helpers["k_grid"].astype(field_dtype),
        linear_power_grid=helpers["linear_power_grid"].astype(field_dtype),
        rfft_factor=helpers["rfft_factor"].astype(field_dtype),
        uv_filter=helpers["uv_filter"].astype(bool),
        noise_sigma=np.asarray(args.flux_noise_sigma, dtype=field_dtype),
        flux_noise_sigma=np.asarray(args.flux_noise_sigma, dtype=field_dtype),
        n2=np.asarray(helpers["n2"], dtype=field_dtype),
        k_nyquist=np.asarray(helpers["k_nyquist"], dtype=field_dtype),
        log_A_true=np.asarray(args.log_a, dtype=field_dtype),
        log_A_init=np.asarray(args.log_a_init, dtype=field_dtype),
        log_A_start=np.asarray(log_a_start, dtype=field_dtype),
        beta_true=np.asarray(args.beta, dtype=field_dtype),
        beta_start=np.asarray(beta_start, dtype=field_dtype),
        rho_floor=np.asarray(args.rho_floor, dtype=field_dtype),
        omega_m_fixed=np.asarray(omega_m, dtype=field_dtype),
        omega_b_fixed=np.asarray(omega_b, dtype=field_dtype),
        sigma8=np.asarray(sigma8, dtype=field_dtype),
        target_seed=np.asarray(args.target_seed, dtype=np.int32),
        initial_seed=np.asarray(args.initial_seed, dtype=np.int32),
        skewer_seed=np.asarray(args.skewer_seed, dtype=np.int32),
        noise_seed=np.asarray(args.noise_seed, dtype=np.int32),
        res=np.asarray(args.res, dtype=np.int32),
        n_skewers=np.asarray(args.n_skewers, dtype=np.int32),
        model_n_steps=np.asarray(DEFAULT_MODEL.n_steps, dtype=np.int32),
        model_boxsize=np.asarray(DEFAULT_MODEL.boxsize, dtype=field_dtype),
        model_a_ini=np.asarray(DEFAULT_MODEL.a_ini, dtype=field_dtype),
        model_a_end=np.asarray(a_end, dtype=field_dtype),
        model_lpt_order=np.asarray(1, dtype=np.int32),
        model_mass_assignment_order=np.asarray(3, dtype=np.int32),
        model_dynamics_model=np.asarray("lpt"),
        model_precision=np.asarray("single"),
    )
    target_bundle_arrays[TARGET_BUNDLE_HASH_KEY] = np.asarray(target_bundle_content_sha256(target_bundle_arrays))

    args.target_bundle_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.target_bundle_output, **target_bundle_arrays)
    save_preview(args.preview_output, delta_z2=delta_z2, clean_flux=clean_flux, noisy_flux=noisy_flux, skewer_xy=skewer_xy)

    print(f"saved target bundle: {args.target_bundle_output}")
    print(f"saved preview: {args.preview_output}")
    print(f"res={args.res}, n_skewers={args.n_skewers}, a_end={a_end:.6f}, worder=3")
    print(f"log_A true/init: {args.log_a:.6e} / {args.log_a_init:.6e}")
    print(f"beta true: {args.beta:.6e}")
    print(f"flux noise sigma: {args.flux_noise_sigma:.6e}")
    print(f"fixed default omega_m: {omega_m:.6e}")
    print(f"fixed default omega_b: {omega_b:.6e}")
    print(f"fixed default sigma8: {sigma8:.6e}")

if __name__ == "__main__":
    main()
