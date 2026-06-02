from __future__ import annotations
import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Any, Self
import numpy as np
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    delta_field: Differentiable[Array[(None, None, None), Float32]] = Field(
        description="3D density contrast field."
    )
    log_A: Differentiable[Float32] = Field(
        description="Log FGPA amplitude."
    )
    beta: Differentiable[Float32] = Field(
        description="FGPA power-law slope."
    )
    skewer_x: Array[(None,), None] = Field(
        description="Zero-based x grid indices for z-axis skewers."
    )
    skewer_y: Array[(None,), None] = Field(
        description="Zero-based y grid indices for z-axis skewers."
    )
    rho_floor: Float32 = Field(
        default=1.0e-3,
        description="Minimum density ratio used inside the FGPA power law.",
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        delta_shape = np.shape(self.delta_field)
        if len(delta_shape) != 3 or len(set(delta_shape)) != 1:
            raise ValueError(f"delta_field must be cubic 3D, got {delta_shape}")
        skewer_x = np.asarray(self.skewer_x)
        skewer_y = np.asarray(self.skewer_y)
        if skewer_x.shape != skewer_y.shape:
            raise ValueError("skewer_x and skewer_y must have matching shapes")
        if isinstance(self.rho_floor, (int, float, np.floating)) and self.rho_floor <= 0.0:
            raise ValueError("rho_floor must be positive")
        return self


class OutputSchema(BaseModel):
    flux_skewers: Differentiable[Array[(None, None), Float32]] = Field(
        description="Transmitted flux skewers with shape (n_skewers, res)."
    )


def _lib_path() -> Path:
    return Path("/tesseract/enzyme/libfgpa_ad.so")


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_lib_path()))
    c_int = ctypes.c_int
    c_float = ctypes.c_float
    c_float_p = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

    lib.fgpa_skewer_forward.argtypes = [
        c_int, c_int, c_int, c_int_p, c_int_p, c_float_p, c_float_p,
        c_float, c_float, c_float,
    ]
    lib.fgpa_skewer_forward.restype = None

    lib.fgpa_skewer_vjp.argtypes = [
        c_int, c_int, c_int, c_int_p, c_int_p,
        c_float_p, c_float_p, c_float_p, c_float_p,
        c_float, ctypes.POINTER(c_float),
        c_float, ctypes.POINTER(c_float),
        c_float,
    ]
    lib.fgpa_skewer_vjp.restype = None

    lib.fgpa_skewer_jvp.argtypes = [
        c_int, c_int, c_int, c_int_p, c_int_p,
        c_float_p, c_float_p, c_float_p, c_float_p,
        c_float, c_float, c_float, c_float, c_float,
    ]
    lib.fgpa_skewer_jvp.restype = None
    return lib


def _as_1d_float32(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1))


def _as_1d_int32(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int32).reshape(-1))


def _prepared_inputs(inputs: InputSchema):
    delta = np.asarray(inputs.delta_field, dtype=np.float32)
    res = int(delta.shape[0])
    skewer_x = _as_1d_int32(inputs.skewer_x)
    skewer_y = _as_1d_int32(inputs.skewer_y)
    if skewer_x.size == 0:
        raise ValueError("at least one skewer is required")
    if np.any(skewer_x < 0) or np.any(skewer_x >= res) or np.any(skewer_y < 0) or np.any(skewer_y >= res):
        raise ValueError("skewer indices are outside the density grid")
    return (
        res,
        int(skewer_x.size),
        res,
        skewer_x,
        skewer_y,
        _as_1d_float32(delta),
        np.float32(inputs.log_A),
        np.float32(inputs.beta),
        np.float32(inputs.rho_floor),
    )


def apply(inputs: InputSchema) -> OutputSchema:
    res, n_skewers, n_los, skewer_x, skewer_y, delta, log_A, beta, rho_floor = _prepared_inputs(inputs)
    flux = np.empty(n_skewers * n_los, dtype=np.float32)
    _lib().fgpa_skewer_forward(
        res, n_skewers, n_los, skewer_x, skewer_y, delta, flux,
        log_A, beta, rho_floor,
    )
    return OutputSchema(flux_skewers=flux.reshape((n_skewers, n_los)))


def abstract_eval(abstract_inputs):
    delta_field = abstract_inputs.delta_field
    delta_shape = delta_field["shape"] if isinstance(delta_field, dict) else delta_field.shape
    skewer_x = abstract_inputs.skewer_x
    skewer_shape = skewer_x["shape"] if isinstance(skewer_x, dict) else skewer_x.shape
    return {
        "flux_skewers": {"shape": (skewer_shape[0], delta_shape[2]), "dtype": "float32"},
    }


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    raise NotImplementedError("Full Jacobian not supported for field inputs.")


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    res, n_skewers, n_los, skewer_x, skewer_y, delta, log_A, beta, rho_floor = _prepared_inputs(inputs)
    ddelta = np.zeros_like(delta)
    if "delta_field" in jvp_inputs:
        ddelta[:] = _as_1d_float32(tangent_vector["delta_field"])
    dlog_A = np.float32(tangent_vector["log_A"]) if "log_A" in jvp_inputs else np.float32(0.0)
    dbeta = np.float32(tangent_vector["beta"]) if "beta" in jvp_inputs else np.float32(0.0)

    flux = np.empty(n_skewers * n_los, dtype=np.float32)
    dflux = np.empty_like(flux)
    _lib().fgpa_skewer_jvp(
        res, n_skewers, n_los, skewer_x, skewer_y,
        delta, ddelta, flux, dflux,
        log_A, dlog_A, beta, dbeta, rho_floor,
    )
    res_dict = {}
    if "flux_skewers" in jvp_outputs:
        res_dict["flux_skewers"] = dflux.reshape((n_skewers, n_los))
    return res_dict


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    res, n_skewers, n_los, skewer_x, skewer_y, delta, log_A, beta, rho_floor = _prepared_inputs(inputs)
    cotangent_flux = np.zeros(n_skewers * n_los, dtype=np.float32)
    if "flux_skewers" in vjp_outputs:
        cotangent_flux[:] = _as_1d_float32(cotangent_vector["flux_skewers"])

    flux = np.empty(n_skewers * n_los, dtype=np.float32)
    _lib().fgpa_skewer_forward(
        res, n_skewers, n_los, skewer_x, skewer_y, delta, flux,
        log_A, beta, rho_floor,
    )
    ddelta = np.zeros_like(delta)
    dlog_A = ctypes.c_float(0.0)
    dbeta = ctypes.c_float(0.0)
    _lib().fgpa_skewer_vjp(
        res, n_skewers, n_los, skewer_x, skewer_y,
        delta, ddelta, flux, cotangent_flux,
        log_A, ctypes.byref(dlog_A),
        beta, ctypes.byref(dbeta),
        rho_floor,
    )

    res_dict = {}
    if "delta_field" in vjp_inputs:
        res_dict["delta_field"] = ddelta.reshape(inputs.delta_field.shape)
    if "log_A" in vjp_inputs:
        res_dict["log_A"] = float(dlog_A.value)
    if "beta" in vjp_inputs:
        res_dict["beta"] = float(dbeta.value)
    return res_dict
