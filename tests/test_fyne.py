import numpy as np
from fyne import blackscholes, heston, wishart
from pytest import mark, raises


def test_blackscholes_impliedvol():
    sigma = 0.2
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    option_price = blackscholes.formula(underlying_price, strike, expiry,
                                        sigma)
    iv = blackscholes.implied_vol(underlying_price, strike, expiry,
                                  option_price)

    assert abs(sigma - iv) < 1e-8

    option_price = blackscholes.formula(underlying_price, strike, expiry,
                                        sigma, True)
    iv = blackscholes.implied_vol(underlying_price, strike, expiry,
                                  option_price, True)

    assert abs(sigma - iv) < 1e-8


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

    option_price = np.array([0.9, 20, np.nan])
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        ivs = blackscholes.implied_vol(underlying_price, strike, expiry,
                                       option_price, assert_no_arbitrage=False)
    assert np.isnan(ivs[0]) and not np.isnan(ivs[1]) and np.isnan(ivs[2])


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

    calibrated_vol_parallel = heston.calibration_vol(
        underlying_price, strikes, expiries, option_prices, kappa, theta, nu,
        rho, put, n_cores=2)

    assert abs(calibrated_vol - calibrated_vol_parallel) < 1e-12


@mark.parametrize('n', [2, 5, 15, 40, 100])
def test_heston_performance(benchmark, n):
    t, v, kappa, a, nu, rho = 0.5, 0.0457, 5.07, 0.2317, 0.48, -0.767
    ks = np.linspace(np.log(0.8), np.log(1.2), n)
    _reduced_formula = np.vectorize(heston._reduced_formula)

    # First execution to trigger JIT
    _reduced_formula(ks, t, v, kappa, a, nu, rho, True)

    benchmark(_reduced_formula, ks, t, v, kappa, a, nu, rho, True)


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


def test_wishart_delta():
    params = dict(
        vol=np.array([[0.0327, 0.0069],
                      [0.0069, 0.0089]]),
        beta=0.6229,
        q=np.array([[0.3193, 0.2590],
                    [0.2899, 0.2469]]),
        m=np.array([[-0.9858, -0.5224],
                    [-0.1288, -0.9746]]),
        r=np.array([[-0.2116, -0.4428],
                    [-0.2113, -0.5921]]),
    )
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    delta_exact = wishart.delta(underlying_price, strike, expiry, **params)
    delta_finite_diffs = 500.*(
        wishart.formula(underlying_price + .001, strike, expiry, **params)
        - wishart.formula(underlying_price - .001, strike, expiry, **params))

    assert abs(delta_exact - delta_finite_diffs) < 1e-3

    delta_exact = wishart.delta(underlying_price, strike, expiry, put=True,
                                **params)
    delta_finite_diffs = 500.*(
        wishart.formula(underlying_price + .001, strike, expiry, put=True,
                        **params)
        - wishart.formula(underlying_price - .001, strike, expiry, put=True,
                          **params))

    assert abs(delta_exact - delta_finite_diffs) < 1e-3


def test_wishart_vega():
    n = 2
    vol = np.array([[0.0327, 0.0069],
                    [0.0069, 0.0089]])
    params = dict(
        beta=0.6229,
        q=np.array([[0.3193, 0.2590],
                    [0.2899, 0.2469]]),
        m=np.array([[-0.9858, -0.5224],
                    [-0.1288, -0.9746]]),
        r=np.array([[-0.2116, -0.4428],
                    [-0.2113, -0.5921]]),
    )
    underlying_price = 100.
    strike = 90.
    expiry = 0.5

    vega_exact = wishart.vega(underlying_price, strike, expiry, vol, **params)
    vega_finite_diffs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h = np.zeros((n, n))
            h[i, j] = 1e-9
            vega_finite_diffs[i, j] = 5e8 * (
                wishart.formula(underlying_price, strike, expiry, vol + h, **params)
                - wishart.formula(underlying_price, strike, expiry, vol - h, **params))

            assert abs(vega_exact[i, j] - vega_finite_diffs[i, j]) < 1e-3
