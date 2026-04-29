import numpy as np

from src.pricing.zcb import (
    vasicek_zcb_price_closed_form,
    zcb_price_monte_carlo_from_paths,
)


def test_zcb_price_at_maturity_is_one():
    price = vasicek_zcb_price_closed_form(
        r0=0.03,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        T=0,
    )

    assert price == 1.0


def test_zcb_price_decreases_when_rate_increases():
    price_low_rate = vasicek_zcb_price_closed_form(
        r0=0.02,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        T=5,
    )

    price_high_rate = vasicek_zcb_price_closed_form(
        r0=0.05,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        T=5,
    )

    assert price_low_rate > price_high_rate


def test_zcb_price_is_positive():
    price = vasicek_zcb_price_closed_form(
        r0=0.03,
        kappa=0.5,
        theta=0.04,
        sigma=0.01,
        T=5,
    )

    assert price > 0


def test_zcb_invalid_maturity():
    try:
        vasicek_zcb_price_closed_form(
            r0=0.03,
            kappa=0.5,
            theta=0.04,
            sigma=0.01,
            T=-1,
        )
        assert False
    except ValueError:
        assert True


def test_monte_carlo_zcb_price_constant_rate():
    rates = np.full((1000, 253), 0.03)
    dt = 1 / 252

    price = zcb_price_monte_carlo_from_paths(rates, dt)

    expected_price = np.exp(-0.03 * 1)

    assert np.isclose(price, expected_price, atol=1e-4)
