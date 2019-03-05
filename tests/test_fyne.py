import numpy as np
from fyne import blackscholes, heston
from pytest import raises


def test_blackscholes_impliedvol():
    sigma = 0.2
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    option_price = blackscholes.formula(underlying_price, strike, expiry,
                                        sigma)
    iv = blackscholes.implied_vol(underlying_price, strike, expiry,
                                  option_price)

    assert abs(sigma - iv) < 1e-6


def test_blackscholes_impliedvol_exception():
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    option_price = 9.
    with raises(ValueError):
        blackscholes.implied_vol(underlying_price, strike, expiry,
                                 option_price)

    option_price = 101.
    with raises(ValueError):
        blackscholes.implied_vol(underlying_price, strike, expiry,
                                 option_price)


def test_blackscholes_delta():
    sigma = 0.2
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    delta_exact = blackscholes.delta(underlying_price, strike, expiry, sigma)
    delta_finite_diffs = 500.*(
        blackscholes.formula(underlying_price + .001, strike, expiry, sigma)
        - blackscholes.formula(underlying_price - .001, strike, expiry, sigma))

    assert abs(delta_exact - delta_finite_diffs) < 1e-3


def test_blackscholes_vega():
    sigma = 0.2
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    vega_exact = blackscholes.vega(underlying_price, strike, expiry, sigma)
    vega_finite_diffs = 500.*(
        blackscholes.formula(underlying_price, strike, expiry, sigma + .001)
        - blackscholes.formula(underlying_price, strike, expiry, sigma - .001))

    assert abs(vega_exact - vega_finite_diffs) < 1e-3


def test_heston_formula():
    vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
    underlying_price = 1640.
    strikes = np.array([1312., 1312., 1640., 1640., 1968., 1968.])
    expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])

    option_prices = np.array([heston.formula(underlying_price, strike, expiry,
                                             vol, kappa, theta, nu, rho)
                              for strike, expiry in zip(strikes, expiries)])
    initial_guess = np.array([vol + 0.01, kappa + 1, theta + 0.01,
                              nu - 0.1, rho - 0.1])
    calibrated = heston.calibration(underlying_price, strikes, expiries,
                                    option_prices, initial_guess)

    original = vol, kappa, theta, nu, rho
    assert np.max(np.abs(np.array(calibrated) - np.array(original))) < 1e-6
