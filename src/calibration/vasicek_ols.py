from __future__ import annotations

import numpy as np


def calibrate_vasicek_ols(
    r: np.ndarray,
    dt: float,
) -> dict:
    """
    Calibrate Vasicek parameters using exact discretization:
        r_{t+dt} = a + b r_t + eps
    where:
        b = exp(-kappa dt)
        a = theta (1 - b)
        Var(eps) = sigma^2 * (1 - exp(-2 kappa dt)) / (2 kappa)

    Inputs:
        r: 1D array of observed short rates (length n)
        dt: time step in years (e.g., 1/252 for daily)

    Returns dict with kappa, theta, sigma, a, b, eps_std
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size < 3:
        raise ValueError("r must be 1D array with length >= 3")

    x = r[:-1]
    y = r[1:]

    # OLS for y = a + b x + e
    X = np.column_stack([np.ones_like(x), x])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    a_hat, b_hat = float(beta_hat[0]), float(beta_hat[1])

    # residuals
    e = y - (a_hat + b_hat * x)
    var_e = float(np.var(e, ddof=2))

    # kappa from b = exp(-kappa dt)
    if b_hat <= 0 or b_hat >= 1:
        # If data is noisy, b_hat may drift; clamp for stability (but report it)
        b_clamped = min(max(b_hat, 1e-6), 1 - 1e-6)
    else:
        b_clamped = b_hat

    kappa_hat = -np.log(b_clamped) / dt
    theta_hat = a_hat / (1.0 - b_clamped)

    # sigma from Var(eps)
    denom = (1.0 - np.exp(-2.0 * kappa_hat * dt)) / (2.0 * kappa_hat)
    sigma_hat = np.sqrt(max(var_e / max(denom, 1e-12), 0.0))

    return {
        "kappa": float(kappa_hat),
        "theta": float(theta_hat),
        "sigma": float(sigma_hat),
        "a": float(a_hat),
        "b": float(b_hat),
        "b_clamped": float(b_clamped),
        "eps_std": float(np.sqrt(max(var_e, 0.0))),
    }
