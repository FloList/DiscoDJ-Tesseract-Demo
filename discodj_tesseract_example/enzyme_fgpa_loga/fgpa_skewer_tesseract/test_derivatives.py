import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract


def reference_fgpa(delta_field, log_A, beta, skewer_x, skewer_y, rho_floor):
    rho = jnp.maximum(1.0 + delta_field[skewer_x, skewer_y, :], rho_floor)
    return jnp.exp(-jnp.exp(log_A) * rho**beta)


def test_fgpa_derivatives():
    t = Tesseract.from_image("fgpa-skewer-tesseract")
    with t:
        res = 8
        key = jax.random.PRNGKey(42)
        delta = 0.15 * jax.random.normal(key, (res, res, res))
        skewer_x = jnp.asarray([0, 2, 4, 6], dtype=jnp.int32)
        skewer_y = jnp.asarray([1, 3, 5, 7], dtype=jnp.int32)
        delta = delta.astype(jnp.float32)
        log_A = jnp.asarray(np.log(0.35), dtype=jnp.float32)
        beta = jnp.asarray(1.6, dtype=jnp.float32)
        rho_floor = jnp.asarray(1.0e-3, dtype=jnp.float32)

        def flux(log_A, beta, delta):
            out = apply_tesseract(
                t,
                inputs=dict(
                    delta_field=delta,
                    log_A=log_A,
                    beta=beta,
                    skewer_x=skewer_x,
                    skewer_y=skewer_y,
                    rho_floor=rho_floor,
                ),
            )
            return out["flux_skewers"]

        def objective(log_A, beta, delta):
            f = flux(log_A, beta, delta)
            return jnp.sum(f**2)

        actual_flux = flux(log_A, beta, delta)
        expected_flux = reference_fgpa(delta, log_A, beta, skewer_x, skewer_y, rho_floor)
        np.testing.assert_allclose(np.asarray(actual_flux), np.asarray(expected_flux), rtol=1.0e-5, atol=1.0e-6)

        eps = jnp.asarray(1.0e-3, dtype=jnp.float32)
        _, grad_log_A = jax.value_and_grad(objective, argnums=0)(log_A, beta, delta)
        fd_log_A = (objective(log_A + eps, beta, delta) - objective(log_A - eps, beta, delta)) / (2 * eps)
        assert abs(float(grad_log_A - fd_log_A)) < 1.0e-3

        _, grad_beta = jax.value_and_grad(objective, argnums=1)(log_A, beta, delta)
        fd_beta = (objective(log_A, beta + eps, delta) - objective(log_A, beta - eps, delta)) / (2 * eps)
        assert abs(float(grad_beta - fd_beta)) < 1.0e-3

        _, grad_delta = jax.value_and_grad(objective, argnums=2)(log_A, beta, delta)
        idx = (2, 3, 4)
        delta_plus = delta.at[idx].add(eps)
        delta_minus = delta.at[idx].add(-eps)
        fd_delta = (objective(log_A, beta, delta_plus) - objective(log_A, beta, delta_minus)) / (2 * eps)
        assert abs(float(grad_delta[idx] - fd_delta)) < 1.0e-3


if __name__ == "__main__":
    test_fgpa_derivatives()
