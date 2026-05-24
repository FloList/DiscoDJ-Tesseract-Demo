# Inspired by
# https://github.com/pasteurlabs/tesseract-core/blob/main/demo/cfd-optimization/cfd-tesseract/tesseract_api.py
from __future__ import annotations
from functools import lru_cache
from typing import Any
# import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from discodj import DiscoDJ
from discodj.core.grids import enforce_hermitian_symmetry
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class ModelParameters(BaseModel):
    n_steps: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of time integration steps, using the "
            "perturbation-theory-informed BullFrog time integrator; see "
            "https://arxiv.org/abs/2409.19049 for details."
        ),
    )
    boxsize: float = Field(
        default=100.0,
        gt=0.0,
        description=(
            "Box size in Mpc/h. For non-cosmologists: 1 pc = 3.26 lightyears, "
            "and h is usually ~0.7, so this simulates cosmic large-scale "
            "structure, i.e. the Cosmic Web, on scales of millions of "
            "lightyears, assuming periodic boundary conditions."
        ),
    )
    a_ini: float = Field(
        default=0.02,
        gt=0.0,
        description=(
            "Scale factor where the initial positions and velocities are set "
            "up with Lagrangian perturbation theory. a = 1 is today; a = 0.02 "
            "means the physical size of the universe back then was 0.02x its "
            "present size in each dimension."
        ),
    )
    a_end: float = Field(
        default=1.0,
        gt=0.0,
        description="Final scale factor where the density field is evaluated.",
    )
    lpt_order: int = Field(
        default=1,
        ge=1,
        le=2,
        description=(
            "Lagrangian perturbation theory order for the initial conditions; "
            "1 is the so-called Zel'dovich approximation."
        ),
    )
    mass_assignment_order: int = Field(
        default=3,
        ge=2,
        le=4,
        description=(
            "Mass assignment order for the particle-mesh-based force "
            "computation: 2 = CIC, 3 = TSC, 4 = PCS."
        ),
    )


class InputSchema(ModelParameters):
    white_noise_fourier: Differentiable[Array[(None, None, None, 2), Float32]] = Field(
        description="Fourier-space white-noise rFFT modes. Last axis stores real and imaginary parts."
    )
    return_trajectory: bool = Field(
        default=False,
        description="Whether to return the full N-body density trajectory for visualization.",
    )
    trajectory_max_frames: int = Field(
        default=12,
        description="Maximum number of density snapshots to return when return_trajectory is true.",
    )


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None, None, None), Float32]] = Field(
        description="Final evolved density contrast."
    )
    trajectory: Differentiable[Array[(None, None, None, None), Float32]] | None = Field(
        default=None,
        description="Optional density trajectory with leading time axis.",
    )
    trajectory_a: Array[(None,), Float32] | None = Field(
        default=None,
        description="Optional scale factors corresponding to trajectory frames.",
    )


def real_white_noise_from_rfft(white_noise_fourier: jax.Array) -> jax.Array:
    res = white_noise_fourier.shape[0]
    modes = white_noise_fourier[..., 0] + 1j * white_noise_fourier[..., 1]
    modes = enforce_hermitian_symmetry(modes, with_jax=True)
    return jnp.fft.irfftn(
        modes,
        s=(res, res, res),
        axes=(0, 1, 2),
        norm="ortho",
    ).astype(jnp.float32)


@lru_cache(maxsize=16)
def build_forward(
    res: int,
    n_steps: int,
    boxsize: float,
    a_ini: float,
    a_end: float,
    lpt_order: int,
    mass_assignment_order: int,
):
    cosmo = dict(
        Omega_c=0.259605,  # cold dark matter content
        Omega_b=0.0488911,  # baryonic matter content
        h=0.67742,  # dimensionless Hubble constant
        n_s=0.96822,  # scalar spectral index of the primordial power spectrum
        sigma8=0.808992,  # amplitude of primordial density fluctuations
    )
    dj = DiscoDJ(
        dim=3,
        res=res,
        name="discodj-tesseract",
        device=jax.default_backend(),
        precision="single",
        boxsize=boxsize,
        cosmo=cosmo,
    )
    # Build timetables for cosmic background evolution
    dj = dj.with_timetables(timetable_settings={"a_max": max(1.0, a_end)})
    # Compute linear power spectrum with Eisenstein & Hu transfer function
    dj = dj.with_linear_ps(transfer_function="Eisenstein-Hu")

    def forward(white_noise_fourier: jax.Array) -> jax.Array:
        # Get real-space white noise field
        white_noise = real_white_noise_from_rfft(white_noise_fourier)
        # Get linear density field
        dj_with_ics = dj.with_ics(
            white_noise_field=white_noise,
            white_noise_space="real",
            sphere_mode=True,  # do not populate corner modes
            try_to_jit=False,
        )
        # Get initial positions and velocities with Lagrangian perturbation theory (LPT)
        dj_with_lpt = dj_with_ics.with_lpt(
            n_order=lpt_order,
            grad_kernel_order=0,
            try_to_jit=False,
        )
        # Run a particle-mesh N-body simulation
        x_final, _, _ = dj_with_lpt.run_nbody(
            a_ini=a_ini,
            a_end=a_end,
            n_steps=n_steps,
            time_var="D",
            stepper="bullfrog",
            method="pm",
            res_pm=res,
            antialias=0,
            grad_kernel_order=0,
            laplace_kernel_order=0,
            nlpt_order_ics=lpt_order,
            worder=mass_assignment_order,
            deconvolve=True,
            n_resample=1,
            return_displacement=False,
            adjoint_method=True,
            collect_all=False,
        )
        # Return density contrast delta = rho/mean(rho) - 1
        return dj_with_lpt.get_delta_from_pos(
            x_final,
            res=res,
            worder=mass_assignment_order,
            deconvolve=False,
            antialias=False,
            try_to_jit=False,
        )

    return jax.jit(forward)


@lru_cache(maxsize=16)
def build_forward_trajectory(
    res: int,
    n_steps: int,
    boxsize: float,
    a_ini: float,
    a_end: float,
    lpt_order: int,
    mass_assignment_order: int,
    trajectory_max_frames: int,
):
    cosmo = dict(
        Omega_c=0.259605,  # cold dark matter content
        Omega_b=0.0488911,  # baryonic matter content
        h=0.67742,  # dimensionless Hubble constant
        n_s=0.96822,  # scalar spectral index of the primordial power spectrum
        sigma8=0.808992,  # amplitude of primordial density fluctuations
    )
    dj = DiscoDJ(
        dim=3,
        res=res,
        name="discodj-tesseract",
        device=jax.default_backend(),
        precision="single",
        boxsize=boxsize,
        cosmo=cosmo,
    )
    # Build timetables for cosmic background evolution
    dj = dj.with_timetables(timetable_settings={"a_max": max(1.0, a_end)})
    # Compute linear power spectrum with Eisenstein & Hu transfer function
    dj = dj.with_linear_ps(transfer_function="Eisenstein-Hu")

    def forward_trajectory(white_noise_fourier: jax.Array) -> tuple[np.ndarray, np.ndarray]:
        # Get real-space white noise field
        white_noise = real_white_noise_from_rfft(white_noise_fourier)
        # Get linear density field
        dj_with_ics = dj.with_ics(
            white_noise_field=white_noise,
            white_noise_space="real",
            sphere_mode=True,  # do not populate corner modes
            try_to_jit=False,
        )
        # Get initial positions and velocities with Lagrangian perturbation theory (LPT)
        dj_with_lpt = dj_with_ics.with_lpt(
            n_order=lpt_order,
            grad_kernel_order=0,
            try_to_jit=False,
        )
        def density_from_pos(x_frame):
            return dj_with_lpt.get_delta_from_pos(
                x_frame,
                res=res,
                worder=mass_assignment_order,
                deconvolve=False,
                antialias=False,
                try_to_jit=False,
            )

        n_saved_frames = min(n_steps + 1, trajectory_max_frames)
        if n_saved_frames == 1:
            selected_frame_indices = {n_steps}
        else:
            selected_frame_indices = set(
                np.linspace(0, n_steps, n_saved_frames, dtype=int).tolist()
            )
        a_trajectory = jnp.linspace(a_ini, a_end, n_steps + 1, dtype=jnp.float32)
        x_current = dj_with_lpt.evaluate_lpt_pos_at_a(a_ini, n_order=lpt_order)
        p_current = dj_with_lpt.evaluate_lpt_psi_dot_at_a(a_ini, n_order=lpt_order)

        density_frames: list[np.ndarray] = []
        scale_factors: list[float] = []

        def append_density_snapshot(frame_index: int, x_frame: jax.Array) -> None:
            density = density_from_pos(x_frame).astype(jnp.float32)
            density_frames.append(
                np.asarray(jax.device_get(density), dtype=np.float32)
            )
            scale_factors.append(
                float(np.asarray(jax.device_get(a_trajectory[frame_index])))
            )
            del density

        if 0 in selected_frame_indices:
            append_density_snapshot(0, x_current)

        for frame_index in range(n_steps):
            step_start = a_trajectory[frame_index]
            step_end = a_trajectory[frame_index + 1]
            dj_step = dj_with_lpt.with_external_ics(
                pos=dj_with_lpt.ensure_flat_shape(x_current),
                vel=dj_with_lpt.ensure_flat_shape(p_current),
            )
            x_current, p_current, _ = dj_step.run_nbody(
                a_ini=step_start,
                a_end=step_end,
                n_steps=1,
                time_var="D",
                stepper="bullfrog",
                method="pm",
                res_pm=res,
                antialias=0,
                grad_kernel_order=0,
                laplace_kernel_order=0,
                nlpt_order_ics=lpt_order,
                worder=mass_assignment_order,
                deconvolve=True,
                n_resample=1,
                return_displacement=False,
                adjoint_method=False,
                collect_all=False,
            )
            output_frame_index = frame_index + 1
            if output_frame_index in selected_frame_indices:
                append_density_snapshot(output_frame_index, x_current)
            del dj_step

        delta_trajectory = np.stack(density_frames, axis=0).astype(np.float32)
        return delta_trajectory, np.asarray(scale_factors, dtype=np.float32)

    return forward_trajectory


def apply_dict(inputs: dict) -> dict:
    white_noise_fourier = inputs["white_noise_fourier"]
    res = int(white_noise_fourier.shape[0])
    if bool(inputs.get("return_trajectory", False)):
        trajectory_max_frames = int(inputs.get("trajectory_max_frames", 12))
        if trajectory_max_frames <= 0:
            raise ValueError("trajectory_max_frames must be positive")
        forward_trajectory = build_forward_trajectory(
            res,
            int(inputs["n_steps"]),
            float(inputs["boxsize"]),
            float(inputs["a_ini"]),
            float(inputs["a_end"]),
            int(inputs["lpt_order"]),
            int(inputs["mass_assignment_order"]),
            trajectory_max_frames,
        )
        trajectory, trajectory_a = forward_trajectory(white_noise_fourier)
        return {
            "result": trajectory[-1],
            "trajectory": trajectory,
            "trajectory_a": trajectory_a,
        }

    forward = build_forward(
        res,
        int(inputs["n_steps"]),
        float(inputs["boxsize"]),
        float(inputs["a_ini"]),
        float(inputs["a_end"]),
        int(inputs["lpt_order"]),
        int(inputs["mass_assignment_order"]),
    )
    delta = forward(white_noise_fourier)
    return {
        "result": delta,
    }


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    return apply_dict(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    if inputs.return_trajectory:
        return apply_dict(inputs.model_dump())
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs):
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    if abstract_inputs.return_trajectory:
        white_noise_shape = abstract_inputs.white_noise_fourier
        if is_shapedtype_dict(white_noise_shape):
            white_noise_shape = white_noise_shape["shape"]
        else:
            white_noise_shape = white_noise_shape.shape
        res = white_noise_shape[0]
        n_trajectory = min(
            int(abstract_inputs.n_steps) + 1,
            int(abstract_inputs.trajectory_max_frames),
        )
        return {
            "result": {"shape": (res, res, res), "dtype": "float32"},
            "trajectory": {
                "shape": (n_trajectory, res, res, res),
                "dtype": "float32",
            },
            "trajectory_a": {"shape": (n_trajectory,), "dtype": "float32"},
        }

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict,
    jvp_inputs: tuple[str],
    jvp_outputs: tuple[str],
    tangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply,
        flatten_with_paths(inputs, include_paths=vjp_inputs),
    )
    return vjp_func(cotangent_vector)[0]
