from __future__ import annotations

import numpy as np


def vasicek_zcb_price_closed_form(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """
    Closed-form ZCB price under Vasicek (risk-neutral parameters assumed).
    P(0,T) = A(T) * exp(-B(T) * r0)
    """
    if T < 0:
        raise ValueError("T must be >= 0")
    if T == 0:
        return 1.0
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    B = (1.0 - np.exp(-kappa * T)) / kappa
    A = np.exp(
        (theta - (sigma**2) / (2.0 * kappa**2)) * (B - T)
        - (sigma**2) * (B**2) / (4.0 * kappa)
    )
    return float(A * np.exp(-B * r0))


def zcb_price_monte_carlo_from_paths(
    rates: np.ndarray,
    dt: float,
) -> float:
    """
    Given simulated short-rate paths r(t), approximate:
      P(0,T) = E[ exp(-∫ r dt) ]
    using Riemann sum for the integral.
    rates shape: (n_paths, n_steps+1)
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    # integral approx on [0,T] using left Riemann sum: sum r_t * dt
    r_left = rates[:, :-1]
    disc = np.exp(-np.sum(r_left, axis=1) * dt)
    return float(np.mean(disc))
