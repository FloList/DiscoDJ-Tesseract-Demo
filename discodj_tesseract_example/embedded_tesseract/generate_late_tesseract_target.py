#!/usr/bin/env python3
"""Generate a late-time tesseract target for the DiscoDJ Tesseract demo.

This script evolves a normal cosmological seed with DiscoDJ, embeds a smooth
straight tesseract wireframe into the evolved density field, and writes the
standard target_bundle.npz schema used by the notebooks. The embedding is done
in positive density space, rho = 1 + delta, so clean and noisy saved targets are
never below delta = -1.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from discodj import DiscoDJ
from scipy.ndimage import gaussian_filter
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from discodj_tesseract_example.common.cosmology import DEFAULT_COSMOLOGY
from discodj_tesseract_example.common.density import (
    add_noise_with_floor,
    delta_from_density,
    density_from_delta,
    log_density_for_plot,
    rms,
)
from discodj_tesseract_example.common.fourier import (
    helpers_from_pk_table,
    make_fourier_white_noise,
    zero_dc_mode,
)
from discodj_tesseract_example.common.target_bundle import save_target_bundle
from target_geometry import (
    add_node_max,
    add_segment_max,
    straight_tesseract_vertices_edges,
)
from tesseract_api import ModelParameters, build_forward
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["savefig.format"] = "pdf"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PARAMS = ModelParameters()


def make_disco_dj(res: int) -> DiscoDJ:
    dj = DiscoDJ(
        dim=3,
        res=res,
        name="late-tesseract-target",
        device=jax.default_backend(),
        precision="single",
        boxsize=DEFAULT_MODEL_PARAMS.boxsize,
        cosmo=DEFAULT_COSMOLOGY,
    )
    return dj.with_timetables(
        timetable_settings={"a_max": max(1.0, DEFAULT_MODEL_PARAMS.a_end)}
    )


def build_pk_table_for_preconditioner(dj: DiscoDJ) -> dict[str, jax.Array]:
    return dj.with_linear_ps(transfer_function="Eisenstein-Hu")._pk_table


def normalize_template(template: np.ndarray) -> np.ndarray:
    template = np.maximum(template.astype(np.float32), 0.0)
    peak = float(np.max(template))
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError("tesseract template has zero/non-finite peak")
    return (template / peak).astype(np.float32)


def make_late_tesseract_template(
    *,
    res: int,
    edge_radius: float | None,
    smoothing_cells: float,
    size_fraction: float,
) -> np.ndarray:
    radius = edge_radius if edge_radius is not None else max(1.25, 0.026 * res)
    vertices, edges = straight_tesseract_vertices_edges(
        res=res,
        size_fraction=size_fraction,
    )

    template = np.zeros((res, res, res), dtype=np.float32)
    for i0, i1 in edges:
        add_segment_max(
            template,
            vertices[i0],
            vertices[i1],
            amplitude=1.0,
            radius=radius,
            support=3.5,
        )
    for vertex in vertices:
        add_node_max(
            template,
            vertex,
            amplitude=0.7,
            radius=1.2 * radius,
            support=3.5,
        )
    if smoothing_cells > 0.0:
        template = gaussian_filter(template, sigma=smoothing_cells, mode="wrap")
    return normalize_template(template)


def embed_tesseract_in_density(
    *,
    base_delta: np.ndarray,
    template: np.ndarray,
    component_rms_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if component_rms_fraction < 0.0:
        raise ValueError("--component-rms-fraction must be non-negative")

    base_rms = rms(base_delta)
    target_component_rms = component_rms_fraction * base_rms
    base_rho = density_from_delta(base_delta)

    def target_for_amplitude(amplitude: float) -> np.ndarray:
        rho = base_rho + np.float32(amplitude) * template
        return delta_from_density(rho)

    lo, hi = 0.0, 1.0
    for _ in range(64):
        component = target_for_amplitude(hi) - base_delta
        if rms(component) >= target_component_rms:
            break
        hi *= 2.0
    else:
        raise RuntimeError("failed to bracket tesseract amplitude")

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        component = target_for_amplitude(mid) - base_delta
        if rms(component) < target_component_rms:
            lo = mid
        else:
            hi = mid

    target = target_for_amplitude(hi)
    delta_shift = (target - base_delta).astype(np.float32)
    density_boost = (np.float32(hi) * template).astype(np.float32)
    return target.astype(np.float32), delta_shift, density_boost, float(hi)


def save_preview(
    path: Path,
    *,
    template: np.ndarray,
    base_target: np.ndarray,
    tesseract_density_boost: np.ndarray,
    target: np.ndarray,
    target_observation: np.ndarray,
    initial_final: np.ndarray,
) -> None:
    z = target.shape[2] // 2
    density_panels = [
        ("tesseract template projection", np.max(template, axis=2)),
        ("tesseract density boost", np.max(tesseract_density_boost, axis=2)),
        ("base evolved target", log_density_for_plot(base_target[:, :, z])),
        ("embedded clean target", log_density_for_plot(target[:, :, z])),
        ("target observation", log_density_for_plot(target_observation[:, :, z])),
        ("initial-seed final", log_density_for_plot(initial_final[:, :, z])),
    ]
    template_vmin, template_vmax = np.percentile(density_panels[0][1], [1.0, 99.5])
    component_abs = np.percentile(np.abs(density_panels[1][1]), 99.0)
    evolved_stack = np.concatenate(
        [
            density_panels[2][1].ravel(),
            density_panels[3][1].ravel(),
            density_panels[4][1].ravel(),
            density_panels[5][1].ravel(),
        ]
    )
    evolved_vmin, evolved_vmax = np.percentile(evolved_stack, [1.0, 99.0])

    fig, axes = plt.subplots(3, 2, figsize=(11.8, 13.0), constrained_layout=True)
    for index, (ax, (title, image)) in enumerate(
        zip(axes.ravel(), density_panels, strict=True)
    ):
        if index == 0:
            kwargs = {"cmap": "viridis", "vmin": template_vmin, "vmax": template_vmax}
        elif index == 1:
            kwargs = {"cmap": "viridis", "vmin": 0.0, "vmax": component_abs}
        else:
            kwargs = {"cmap": "magma", "vmin": evolved_vmin, "vmax": evolved_vmax}
        plot = ax.imshow(image.T, origin="lower", **kwargs)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, shrink=0.85)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=123)
    parser.add_argument("--initial-seed", type=int, default=17)
    parser.add_argument("--noise-sigma", type=float, default=1.0)
    parser.add_argument(
        "--component-rms-fraction",
        type=float,
        default=0.7,
        help="RMS of late-time tesseract component as a fraction of base evolved target RMS.",
    )
    parser.add_argument("--edge-radius", type=float, default=None)
    parser.add_argument("--smoothing-cells", type=float, default=0.9)
    parser.add_argument("--size-fraction", type=float, default=0.76)
    parser.add_argument(
        "--target-bundle-output",
        type=Path,
        default=HERE / "target_bundle.npz",
        help="Compressed NPZ containing target, observation, modes, and helpers.",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=HERE / "late_tesseract_target.pdf",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.res <= 0:
        raise ValueError("--res must be positive")
    if args.noise_sigma < 0.0:
        raise ValueError("--noise-sigma must be non-negative")
    if args.component_rms_fraction < 0.0:
        raise ValueError("--component-rms-fraction must be non-negative")
    if args.smoothing_cells < 0.0:
        raise ValueError("--smoothing-cells must be non-negative")
    if args.size_fraction <= 0.0:
        raise ValueError("--size-fraction must be positive")
    if args.edge_radius is not None and args.edge_radius <= 0.0:
        raise ValueError("--edge-radius must be positive when provided")
    if args.target_seed == args.initial_seed:
        raise ValueError("target seed and initial seed must differ")


def main() -> None:
    args = parse_args()
    validate_args(args)

    dj = make_disco_dj(args.res)
    forward = build_forward(
        args.res,
        DEFAULT_MODEL_PARAMS.n_steps,
        DEFAULT_MODEL_PARAMS.boxsize,
        DEFAULT_MODEL_PARAMS.a_ini,
        DEFAULT_MODEL_PARAMS.a_end,
        DEFAULT_MODEL_PARAMS.lpt_order,
        DEFAULT_MODEL_PARAMS.mass_assignment_order,
    )
    helpers = helpers_from_pk_table(
        pk_table=build_pk_table_for_preconditioner(dj),
        res=args.res,
        boxsize=DEFAULT_MODEL_PARAMS.boxsize,
        noise_sigma=args.noise_sigma,
    )

    target_base_white_noise = zero_dc_mode(make_fourier_white_noise(args.target_seed, args.res))
    initial_white_noise = zero_dc_mode(make_fourier_white_noise(args.initial_seed, args.res))

    base_target = np.asarray(
        forward(jnp.asarray(target_base_white_noise)),
        dtype=np.float32,
    )
    initial_final = np.asarray(
        forward(jnp.asarray(initial_white_noise)),
        dtype=np.float32,
    )
    template = make_late_tesseract_template(
        res=args.res,
        edge_radius=args.edge_radius,
        smoothing_cells=args.smoothing_cells,
        size_fraction=args.size_fraction,
    )
    (
        target,
        tesseract_delta_shift,
        tesseract_density_boost,
        tesseract_amplitude,
    ) = embed_tesseract_in_density(
        base_delta=base_target,
        template=template,
        component_rms_fraction=args.component_rms_fraction,
    )
    target_observation = add_noise_with_floor(
        target,
        seed=args.target_seed + 100_000,
        noise_sigma=float(helpers["noise_sigma"]),
    )

    args.target_bundle_output.parent.mkdir(parents=True, exist_ok=True)
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
    save_target_bundle(
        args.target_bundle_output,
        target=target.astype(np.float32),
        target_observation=target_observation.astype(np.float32),
        initial_white_noise_fourier=initial_white_noise.astype(np.float32),
        target_base_white_noise_fourier=target_base_white_noise.astype(np.float32),
        target_cosmological_base=base_target.astype(np.float32),
        target_tesseract_template=template.astype(np.float32),
        target_tesseract_density_boost=tesseract_density_boost.astype(np.float32),
        k_grid=helpers["k_grid"].astype(np.float32),
        linear_power_grid=helpers["linear_power_grid"].astype(np.float32),
        rfft_factor=helpers["rfft_factor"].astype(np.float32),
        uv_filter=helpers["uv_filter"].astype(bool),
        noise_sigma=np.asarray(helpers["noise_sigma"], dtype=np.float32),
        n2=np.asarray(helpers["n2"], dtype=np.float32),
        k_nyquist=np.asarray(helpers["k_nyquist"], dtype=np.float32),
        component_rms_fraction=np.asarray(args.component_rms_fraction, dtype=np.float32),
        tesseract_amplitude=np.asarray(tesseract_amplitude, dtype=np.float32),
        smoothing_cells=np.asarray(args.smoothing_cells, dtype=np.float32),
        size_fraction=np.asarray(args.size_fraction, dtype=np.float32),
        target_seed=np.asarray(args.target_seed, dtype=np.int32),
        initial_seed=np.asarray(args.initial_seed, dtype=np.int32),
        res=np.asarray(args.res, dtype=np.int32),
    )
    save_preview(
        args.preview_output,
        template=template,
        base_target=base_target,
        tesseract_density_boost=tesseract_density_boost,
        target=target,
        target_observation=target_observation,
        initial_final=initial_final,
    )

    print(f"saved target bundle: {args.target_bundle_output}")
    print(f"saved preview: {args.preview_output}")
    print(f"noise sigma: {float(helpers['noise_sigma']):.6e}")
    print(f"component RMS fraction: {args.component_rms_fraction:.6e}")
    print(f"tesseract amplitude: {tesseract_amplitude:.6e}")
    print(
        "base target stats: "
        f"mean={base_target.mean():.6e}, std={base_target.std():.6e}, "
        f"min={base_target.min():.6e}, max={base_target.max():.6e}"
    )
    print(
        "embedded target stats: "
        f"mean={target.mean():.6e}, std={target.std():.6e}, "
        f"min={target.min():.6e}, max={target.max():.6e}"
    )
    print(
        "target observation stats: "
        f"mean={target_observation.mean():.6e}, std={target_observation.std():.6e}, "
        f"min={target_observation.min():.6e}, max={target_observation.max():.6e}"
    )
    print(
        "component stats: "
        f"mean={tesseract_delta_shift.mean():.6e}, "
        f"std={tesseract_delta_shift.std():.6e}"
    )


if __name__ == "__main__":
    main()
