# Inspired by
# https://github.com/pasteurlabs/tesseract-core/blob/main/demo/cfd-optimization/cfd-tesseract/tesseract_api.py
# This Tesseract evolves an externally supplied linear density field with DiscoDJ.
# In the demo notebook, derivatives are taken with respect to the supplied field;
# cosmological parameters are fixed background settings.
from __future__ import annotations
from functools import lru_cache
from typing import Any, Literal
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from discodj import DiscoDJ
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable
from tesseract_core.runtime import Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

FIXED_OMEGA_M = 0.3084961
FIXED_OMEGA_B = 0.0488911
FIXED_H = 0.67742
FIXED_N_S = 0.96822
FIXED_SIGMA8 = 0.808992


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
    dynamics_model: Literal["nbody", "lpt"] = Field(
        default="nbody",
        description=(
            "Forward dynamics model. 'nbody' runs the particle-mesh N-body "
            "solver; 'lpt' evaluates the configured LPT/Zel'dovich positions "
            "directly at a_end."
        ),
    )
    precision: Literal["single"] = Field(
        default="single",
        description="Floating-point precision used internally by DiscoDJ.",
    )


class InputSchema(ModelParameters):
    delta_ini: Differentiable[Array[(None, None, None), None]] = Field(
        description="Externally supplied linear density contrast on the simulation grid."
    )
    omega_m: Float32 = Field(
        default=FIXED_OMEGA_M,
        description="Fixed total matter density fraction Omega_m.",
    )
    omega_b: Float32 = Field(
        default=FIXED_OMEGA_B,
        description="Fixed baryon density fraction Omega_b.",
    )
    h: Float32 = Field(
        default=FIXED_H,
        description="Fixed dimensionless Hubble parameter.",
    )
    n_s: Float32 = Field(
        default=FIXED_N_S,
        description="Fixed scalar spectral index.",
    )
    sigma8: Float32 = Field(
        default=FIXED_SIGMA8,
        description="Fixed amplitude setting passed to DiscoDJ.",
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        delta_shape = np.shape(self.delta_ini)
        if len(delta_shape) != 3 or len(set(delta_shape)) != 1:
            raise ValueError(f"delta_ini must be cubic 3D, got {delta_shape}")

        abstract_values = (
            not isinstance(value, (int, float, np.floating))
            for value in (self.omega_m, self.omega_b, self.h, self.n_s, self.sigma8)
        )
        if any(abstract_values):
            return self
        if self.omega_b <= 0.0:
            raise ValueError("omega_b must be positive.")
        if self.omega_m <= self.omega_b:
            raise ValueError("omega_m must be larger than omega_b.")
        if self.h <= 0.0:
            raise ValueError("h must be positive.")
        if self.sigma8 <= 0.0:
            raise ValueError("sigma8 must be positive.")
        fixed_values = {
            "omega_m": (self.omega_m, FIXED_OMEGA_M),
            "omega_b": (self.omega_b, FIXED_OMEGA_B),
            "h": (self.h, FIXED_H),
            "n_s": (self.n_s, FIXED_N_S),
            "sigma8": (self.sigma8, FIXED_SIGMA8),
        }
        for name, (value, expected) in fixed_values.items():
            if not np.isclose(value, expected):
                raise ValueError(f"{name} is fixed at {expected}.")
        return self


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None, None, None), None]] = Field(
        description="Final evolved density contrast."
    )


@lru_cache(maxsize=16)
def build_forward(
    res: int,
    n_steps: int,
    boxsize: float,
    a_ini: float,
    a_end: float,
    lpt_order: int,
    mass_assignment_order: int,
    dynamics_model: str,
    precision: str,
):
    if dynamics_model not in {"nbody", "lpt"}:
        raise ValueError("dynamics_model must be one of {'nbody', 'lpt'}")
    if precision != "single":
        raise ValueError("precision must be 'single'")
    field_dtype = jnp.float32

    def forward(
        delta_ini: jax.Array,
        omega_m: jax.Array,
    ) -> jax.Array:
        # Build the fixed DiscoDJ background used by this demo.
        omega_m = jnp.asarray(omega_m)
        cosmo = dict(
            Omega_c=omega_m - jnp.asarray(FIXED_OMEGA_B, dtype=omega_m.dtype),
            Omega_b=FIXED_OMEGA_B,
            h=FIXED_H,
            n_s=FIXED_N_S,
            sigma8=FIXED_SIGMA8,
        )
        dj = DiscoDJ(
            dim=3,
            res=res,
            name="discodj-external-ics-forward",
            device=jax.default_backend(),
            precision=precision,
            boxsize=boxsize,
            cosmo=cosmo,
        )
        dj = dj.with_timetables(timetable_settings={"a_max": max(1.0, a_end)})
        # Use the externally supplied linear density contrast as the initial field.
        dj_with_ics = dj.with_external_ics(delta=delta_ini.astype(field_dtype))
        # Get initial positions and velocities with Lagrangian perturbation theory (LPT)
        dj_with_lpt = dj_with_ics.with_lpt(
            n_order=lpt_order,
            grad_kernel_order=0,
            try_to_jit=False,
        )
        if dynamics_model == "lpt":
            x_final = dj_with_lpt.evaluate_lpt_pos_at_a(a_end, n_order=lpt_order)
        else:
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
        ).astype(field_dtype)

    return jax.jit(forward)


def apply_dict(inputs: dict) -> dict:
    delta_ini = inputs["delta_ini"]
    res = int(delta_ini.shape[0])
    forward = build_forward(
        res,
        int(inputs["n_steps"]),
        float(inputs["boxsize"]),
        float(inputs["a_ini"]),
        float(inputs["a_end"]),
        int(inputs["lpt_order"]),
        int(inputs["mass_assignment_order"]),
        str(inputs.get("dynamics_model", "nbody")),
        str(inputs.get("precision", "single")),
    )
    return {
        "result": forward(
            delta_ini,
            inputs["omega_m"],
        )
    }


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    return apply_dict(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    outputs = apply_jit(inputs.model_dump())
    return OutputSchema(
        result=np.asarray(jax.device_get(outputs["result"]), dtype=np.float32)
    )


def abstract_eval(abstract_inputs):
    delta_ini = abstract_inputs.delta_ini
    shape = delta_ini["shape"] if isinstance(delta_ini, dict) else delta_ini.shape
    return {"result": {"shape": shape, "dtype": "float32"}}


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
