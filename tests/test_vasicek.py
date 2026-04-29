import numpy as np

from src.models.vasicek import simulate_vasicek_exact, simulate_vasicek_euler


def test_vasicek_exact_shape():
    rates = simulate_vasicek_exact(
        r0=0.03,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        dt=1 / 252,
        n_steps=252,
        n_paths=100,
        seed=42,
    )

    assert isinstance(rates, np.ndarray)
    assert rates.shape == (100, 253)


def test_vasicek_euler_shape():
    rates = simulate_vasicek_euler(
        r0=0.03,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        dt=1 / 252,
        n_steps=252,
        n_paths=100,
        seed=42,
    )

    assert isinstance(rates, np.ndarray)
    assert rates.shape == (100, 253)


def test_vasicek_initial_rate_is_correct():
    rates = simulate_vasicek_exact(
        r0=0.03,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        dt=1 / 252,
        n_steps=252,
        n_paths=100,
        seed=42,
    )

    assert np.allclose(rates[:, 0], 0.03)


def test_vasicek_invalid_parameters():
    try:
        simulate_vasicek_exact(
            r0=0.03,
            kappa=-0.5,
            theta=0.04,
            sigma=0.01,
            dt=1 / 252,
            n_steps=252,
            n_paths=100,
            seed=42,
        )
        assert False
    except ValueError:
        assert True
