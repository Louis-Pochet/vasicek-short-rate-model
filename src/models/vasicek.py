from __future__ import annotations

import numpy as np


def simulate_vasicek_exact(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Exact discretization of Vasicek (Ornstein-Uhlenbeck) under:
        dr_t = kappa (theta - r_t) dt + sigma dW_t

    Returns:
        rates: shape (n_paths, n_steps+1)
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive")

    rng = np.random.default_rng(seed)

    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = r0

    exp_kdt = np.exp(-kappa * dt)
    mean_coef = exp_kdt
    const = theta * (1.0 - exp_kdt)

    var = (sigma**2) * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
    std = np.sqrt(max(var, 0.0))

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        rates[:, t + 1] = const + mean_coef * rates[:, t] + std * z

    return rates


def simulate_vasicek_euler(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Euler-Maruyama discretization (simple baseline).
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    rng = np.random.default_rng(seed)
    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = r0

    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        rates[:, t + 1] = (
            rates[:, t]
            + kappa * (theta - rates[:, t]) * dt
            + sigma * sqrt_dt * z
        )
    return rates
