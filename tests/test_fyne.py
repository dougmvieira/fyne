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

    option_price = blackscholes.formula(underlying_price, strike, expiry,
                                        sigma, True)
    iv = blackscholes.implied_vol(underlying_price, strike, expiry,
                                  option_price, True)

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

    delta_exact = blackscholes.delta(underlying_price, strike, expiry, sigma,
                                     True)
    delta_finite_diffs = 500.*(
        blackscholes.formula(underlying_price + .001, strike, expiry, sigma,
                             True)
        - blackscholes.formula(underlying_price - .001, strike, expiry, sigma,
                               True))

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

    vega_finite_diffs = 500.*(
        blackscholes.formula(underlying_price, strike, expiry, sigma + .001,
                             True)
        - blackscholes.formula(underlying_price, strike, expiry, sigma - .001,
                               True))

    assert abs(vega_exact - vega_finite_diffs) < 1e-3


def test_heston_calibration_crosssectional():
    vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
    underlying_price = 100.
    strikes = np.array([80., 80., 100., 100., 120., 120.])
    expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    put = np.array([False, False, False, False, True, True])

    option_prices = heston.formula(underlying_price, strikes, expiries, vol,
                                   kappa, theta, nu, rho, put)
    initial_guess = np.array([vol + 0.01, kappa + 1, theta + 0.01,
                              nu - 0.1, rho - 0.1])
    calibrated = heston.calibration_crosssectional(
        underlying_price, strikes, expiries, option_prices, initial_guess, put)

    original = vol, kappa, theta, nu, rho
    assert np.max(np.abs(np.array(calibrated) - np.array(original))) < 1e-6


def test_heston_calibration_panel():
    kappa, theta, nu, rho = 5.07, 0.0457, 0.48, -0.767
    underlying_prices = np.array([90., 100., 95.])
    vols = np.array([0.05, 0.045, 0.055])
    strikes = np.array([80., 80., 100., 100., 120., 120.])
    expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    put = np.array([False, False, False, False, True, True])

    option_prices = (
        heston.formula(underlying_prices[:, None], strikes, expiries,
                       vols[:, None], kappa, theta, nu, rho, put))

    initial_guess = np.array([vols[1] + 0.01, kappa + 1, theta + 0.01,
                              nu - 0.1, rho - 0.1])
    calibrated_tuple = (
        heston.calibration_panel(underlying_prices, strikes, expiries,
                                 option_prices, initial_guess, put))

    calibrated = np.concatenate((calibrated_tuple[0], calibrated_tuple[1:]))
    original = np.concatenate((vols, (kappa, theta, nu, rho)))

    assert np.max(np.abs(np.array(calibrated) - np.array(original))) < 1e-6


def test_heston_calibration_vol():
    vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
    underlying_price = 100.
    strikes = np.array([80., 80., 100., 100., 120., 120.])
    expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    put = np.array([False, False, False, False, True, True])

    option_prices = heston.formula(underlying_price, strikes, expiries, vol,
                                   kappa, theta, nu, rho, put)
    calibrated_vol = heston.calibration_vol(
        underlying_price, strikes, expiries, option_prices, kappa, theta, nu,
        rho, put)

    assert abs(vol - calibrated_vol) < 1e-6


def test_heston_benchmark():
    assert heston.benchmark(1) > 0


def test_heston_delta():
    params = 0.0457, 5.07, 0.0457, 0.48, -0.767
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    delta_exact = heston.delta(underlying_price, strike, expiry, *params)
    delta_finite_diffs = 500.*(
        heston.formula(underlying_price + .001, strike, expiry, *params)
        - heston.formula(underlying_price - .001, strike, expiry, *params))

    assert abs(delta_exact - delta_finite_diffs) < 1e-3

    delta_exact = heston.delta(underlying_price, strike, expiry, *params,
                               put=True)
    delta_finite_diffs = 500.*(
        heston.formula(underlying_price + .001, strike, expiry, *params,
                       put=True)
        - heston.formula(underlying_price - .001, strike, expiry, *params,
                         put=True))

    assert abs(delta_exact - delta_finite_diffs) < 1e-3


def test_heston_vega():
    vol = 0.0457
    params = 5.07, 0.0457, 0.48, -0.767
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    vega_exact = heston.vega(underlying_price, strike, expiry, vol, *params)
    vega_finite_diffs = 500.*(
        heston.formula(underlying_price, strike, expiry, vol + .001, *params)
        - heston.formula(underlying_price, strike, expiry, vol - .001, *params))

    assert abs(vega_exact - vega_finite_diffs) < 1e-3

    vega_exact = heston.vega(underlying_price, strike, expiry, vol, *params)
    vega_finite_diffs = 500.*(
        heston.formula(underlying_price, strike, expiry, vol + .001, *params,
                       put=True)
        - heston.formula(underlying_price, strike, expiry, vol - .001, *params,
                         put=True))

    assert abs(vega_exact - vega_finite_diffs) < 1e-3
