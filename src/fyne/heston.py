import cmath
from concurrent.futures import ProcessPoolExecutor
from ctypes import c_void_p
from itertools import repeat
from math import pi

import numpy as np
from numba import carray, cfunc, njit
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.optimize import leastsq

from fyne import common
from fyne import blackscholes


def formula(underlying_price, strike, expiry, vol, kappa, theta, nu, rho,
            put=False, assert_no_arbitrage=False):
    r"""Heston formula

    Computes the price of the option according to the Heston formula.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    vol : float
        Instantaneous volatility.
    kappa : float
        Model parameter :math:`\kappa`.
    theta : float
        Model parameter :math:`\theta`.
    nu : float
        Model parameter :math:`\nu`.
    rho : float
        Model parameter :math:`\rho`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.

    Returns
    -------
    float
        Option price according to Heston formula.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import heston
    >>> v, kappa, theta, nu, rho = 0.2, 1.3, 0.04, 0.4, -0.3
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> expiry = 0.5
    >>> call_price = heston.formula(underlying_price, strike, expiry, v, kappa,
    ...                             theta, nu, rho)
    >>> np.round(call_price, 2)
    16.32
    >>> put_price = heston.formula(underlying_price, strike, expiry, v, kappa,
    ...                            theta, nu, rho, put=True)
    >>> np.round(put_price, 2)
    6.32

    """

    ks = np.log(strike/underlying_price)
    a = kappa*theta
    broadcasted = np.broadcast(ks, expiry, vol)
    call = np.empty(broadcasted.shape)
    call.flat = [
        _reduced_formula(k, t, v, kappa, a, nu, rho, assert_no_arbitrage)
        for (k, t, v) in broadcasted
    ]
    call *= underlying_price
    return common._put_call_parity(call, underlying_price, strike, put)


def delta(underlying_price, strike, expiry, vol, kappa, theta, nu, rho,
          put=False):
    r"""Heston Greek delta

    Computes the Greek :math:`\Delta` (delta) of the option according to the
    Heston formula.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    vol : float
        Instantaneous volatility.
    kappa : float
        Model parameter :math:`\kappa`.
    theta : float
        Model parameter :math:`\theta`.
    nu : float
        Model parameter :math:`\nu`.
    rho : float
        Model parameter :math:`\rho`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.

    Returns
    -------
    float
        Option Greek :math:`\Delta` (delta) according to Heston formula.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import heston
    >>> v, kappa, theta, nu, rho = 0.2, 1.3, 0.04, 0.4, -0.3
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> call_delta = heston.delta(underlying_price, strike, maturity, v, kappa,
    ...                           theta, nu, rho)
    >>> np.round(call_delta, 2)
    0.72
    >>> put_delta = heston.delta(underlying_price, strike, maturity, v, kappa,
    ...                      theta, nu, rho, put=True)
    >>> np.round(put_delta, 2)
    -0.28

    """

    k = np.log(strike/underlying_price)
    a = kappa*theta
    call_delta = _reduced_delta(k, expiry, vol, kappa, a, nu, rho)
    return common._put_call_parity_delta(call_delta, put)


def vega(underlying_price, strike, expiry, vol, kappa, theta, nu, rho):
    r"""Heston Greek vega

    Computes the Greek :math:`\mathcal{V}` (vega) of the option according to
    the Heston formula.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    vol : float
        Instantaneous volatility.
    kappa : float
        Model parameter :math:`\kappa`.
    theta : float
        Model parameter :math:`\theta`.
    nu : float
        Model parameter :math:`\nu`.
    rho : float
        Model parameter :math:`\rho`.

    Returns
    -------
    float
        Option Greek :math:`\mathcal{V}` (vega) according to Heston formula.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import heston
    >>> v, kappa, theta, nu, rho = 0.2, 1.3, 0.04, 0.4, -0.3
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> vega = heston.vega(underlying_price, strike, maturity, v, kappa, theta,
    ...                    nu, rho)
    >>> np.round(vega, 2)
    22.5

    """

    k = np.log(strike/underlying_price)
    a = kappa*theta
    return _reduced_vega(k, expiry, vol, kappa, a, nu, rho)*underlying_price


def calibration_crosssectional(underlying_price, strikes, expiries,
                               option_prices, initial_guess, put=False,
                               weights=None):
    r"""Heston cross-sectional calibration

    Recovers the Heston model parameters from options prices at a single point
    in time. The calibration is performed using the Levenberg-Marquardt
    algorithm.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strikes : numpy.array
        One-dimensional array of option strikes. Must be of the same length as
        the expiries and option_prices arrays.
    expiries : numpy.array
        One-dimensional array of option expiries. The expiries are the time
        remaining until the expiry of the option. Must be of the same length as
        the strikes and option_prices arrays.
    option_prices : numpy.array
        One-dimensional array of call options prices. Must be of the same
        length as the expiries and strikes arrays.
    initial_guess : tuple
        Initial guess for instantaneous volatility :math:`V_0` as :obj:float
        and the Heston parameters :math:`\kappa`, :math:`\theta`, :math:`\nu`
        and :math:`\rho`, respectively, as :obj:float.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.
    weights : numpy.array, optional
        One-dimensional array of call options prices. Must be of the same
        length as the option_prices, expiries and strikes arrays.

    Returns
    -------
    tuple
        Returns the calibrated instantaneous volatility :math:`V_0` and the
        Heston parameters :math:`\kappa`, :math:`\theta`, :math:`\nu` and
        :math:`\rho`, respectively, as :obj:`float`.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import heston
    >>> vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
    >>> underlying_price = 1640.
    >>> strikes = np.array([1312., 1312., 1640., 1640., 1968., 1968.])
    >>> expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    >>> put = np.array([False, False, False, False, True, True])

    >>> option_prices = heston.formula(underlying_price, strikes, expiries,
    ...                                vol, kappa, theta, nu, rho, put)
    >>> initial_guess = np.array([vol + 0.01, kappa + 1, theta + 0.01,
    ...                           nu - 0.1, rho - 0.1])
    >>> calibrated = heston.calibration_crosssectional(
    ...     underlying_price, strikes, expiries, option_prices, initial_guess,
    ...     put)
    >>> [np.round(param, 4) for param in calibrated]
    [0.0457, 5.07, 0.0457, 0.48, -0.767]

    """

    calls = common._put_call_parity_reverse(option_prices, underlying_price,
                                            strikes, put)
    cs = calls/underlying_price
    ks = np.log(strikes/underlying_price)
    ws = 1/cs if weights is None else weights/cs
    vol0, kappa0, theta0, nu0, rho0 = initial_guess
    params = np.array([vol0, kappa0, kappa0*theta0, nu0, rho0])

    vol, kappa, a, nu, rho = _reduced_calib_xsect(cs, ks, expiries, ws, params)

    return vol, kappa, a/kappa, nu, rho


def calibration_panel(underlying_prices, strikes, expiries, option_prices,
                      initial_guess, put=False, weights=None):
    r"""Heston panel calibration

    Recovers the Heston model parameters from options prices across strikes,
    maturities and time. The calibration is performed using the
    Levenberg-Marquardt algorithm.

    Parameters
    ----------
    underlying_price : numpy.array
        One-dimensional array of prices of the underlying asset at each point
        in time.
    strikes : numpy.array
        One-dimensional array of option strikes. Must be of the same length as
        the expiries array.
    expiries : numpy.array
        One-dimensional array of option expiries. The expiries are the time
        remaining until the expiry of the option. Must be of the same length as
        the strikes array.
    option_prices : numpy.array
        Two-dimensional array of the call options prices. The array must be
        :math:`n`-by-:math:`d`, where :math:`n` is the size of
        `underlying_price` and :math:`d` is the size of `strikes` or
        `expiries`.
    initial_guess : tuple
        Initial guess for instantaneous volatility :math:`V_0` as :obj:float
        and the Heston parameters :math:`\kappa`, :math:`\theta`, :math:`\nu`
        and :math:`\rho`, respectively, as :obj:float.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.
    weights : numpy.array, optional
        One-dimensional array of call options prices. Must be of the same
        length as the option_prices, expiries and strikes arrays.

    Returns
    -------
    tuple
        Returns the calibrated instantaneous volatilities :math:`V_0` as a
        :obj:`numpy.array` and the Heston parameters :math:`\kappa`,
        :math:`\theta`, :math:`\nu` and :math:`\rho`, respectively, as
        :obj:`float`.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import heston
    >>> kappa, theta, nu, rho = 5.07, 0.0457, 0.48, -0.767
    >>> underlying_prices = np.array([90., 100., 95.])
    >>> vols = np.array([0.05, 0.045, 0.055])
    >>> strikes = np.array([80., 80., 100., 100., 120., 120.])
    >>> expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    >>> put = np.array([False, False, False, False, True, True])
    >>> option_prices = (
    ...     heston.formula(underlying_prices[:, None], strikes, expiries,
    ...                    vols[:, None], kappa, theta, nu, rho, put))
    >>> initial_guess = np.array([vols[1] + 0.01, kappa + 1, theta + 0.01,
    ...                           nu - 0.1, rho - 0.1])
    >>> vols, kappa, theta, nu, rho = heston.calibration_panel(
    ...     underlying_prices, strikes, expiries, option_prices, initial_guess,
    ...     put)
    >>> np.round(vols, 4)
    array([0.05 , 0.045, 0.055])
    >>> [np.round(param, 4) for param in (kappa, theta, nu, rho)]
    [5.07, 0.0457, 0.48, -0.767]

    """

    calls = common._put_call_parity_reverse(
        option_prices, underlying_prices[:, None], strikes, put)
    cs = calls/underlying_prices[:, None]
    ks = np.log(strikes[None, :]/underlying_prices[:, None])
    ws = 1/cs if weights is None else weights/cs
    vol0, kappa0, theta0, nu0, rho0 = initial_guess
    params = vol0*np.ones(len(underlying_prices) + 4)
    params[-4:] = kappa0, kappa0*theta0, nu0, rho0

    calibrated = _reduced_calib_panel(cs, ks, expiries, ws, params)
    vols = calibrated[:-4]
    kappa, a, nu, rho = calibrated[-4:]

    return vols, kappa, a/kappa, nu, rho


def calibration_vol(underlying_price, strikes, expiries, option_prices, kappa,
                    theta, nu, rho, put=False, vol_guess=0.1, weights=None,
                    n_cores=None):
    r"""Heston volatility calibration

    Recovers the Heston instantaneous volatility from options prices at a
    single point in time. The Heston model parameters must be provided. The
    calibration is performed using the Levenberg-Marquardt algorithm.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strikes : numpy.array
        One-dimensional array of option strikes. Must be of the same length as
        the expiries and option_prices arrays.
    expiries : numpy.array
        One-dimensional array of option expiries. The expiries are the time
        remaining until the expiry of the option. Must be of the same length as
        the strikes and option_prices arrays.
    option_prices : numpy.array
        One-dimensional array of call options prices. Must be of the same
        length as the expiries and strikes arrays.
    kappa : float
        Model parameter :math:`\kappa`.
    theta : float
        Model parameter :math:`\theta`.
    nu : float
        Model parameter :math:`\nu`.
    rho : float
        Model parameter :math:`\rho`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.
    vol_guess : float, optional
        Initial guess for instantaneous volatility :math:`V_0`. Defaults to
        0.1.
    weights : numpy.array, optional
        One-dimensional array of call options prices. Must be of the same
        length as the option_prices, expiries and strikes arrays.

    Returns
    -------
    float
        Returns the calibrated instantaneous volatility :math:`V_0`.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import heston
    >>> vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
    >>> underlying_price = 1640.
    >>> strikes = np.array([1312., 1312., 1640., 1640., 1968., 1968.])
    >>> expiries = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    >>> put = np.array([False, False, False, False, True, True])
    >>> option_prices = heston.formula(underlying_price, strikes, expiries, vol,
    ...                                kappa, theta, nu, rho, put)
    >>> calibrated_vol = heston.calibration_vol(
    ...     underlying_price, strikes, expiries, option_prices, kappa, theta,
    ...     nu, rho, put)
    >>> np.round(calibrated_vol, 4)
    0.0457

    """

    calls = common._put_call_parity_reverse(option_prices, underlying_price,
                                            strikes, put)
    cs = calls/underlying_price
    ks = np.log(strikes/underlying_price)
    ws = 1/cs if weights is None else weights/cs

    vol, = _reduced_calib_vol(cs, ks, expiries, ws, kappa, kappa*theta, nu,
                              rho, np.array([vol_guess]), n_cores=n_cores)

    return vol


@njit
def _heston_psi(u, t, kappa, a, nu, rho):
    d = cmath.sqrt(nu**2*(u**2 + 1j*u) + (kappa - 1j*nu*rho*u)**2)
    g = (-d + kappa - 1j*nu*rho*u)/(d + kappa - 1j*nu*rho*u)
    h = (g*cmath.exp(-d*t) - 1)/(g - 1)
    psi_1 = a*(t*(-d + kappa - 1j*nu*rho*u) - 2*cmath.log(h))/nu**2
    psi_2 = (1 - cmath.exp(-d*t))*(-d + kappa - 1j*nu*rho*u)/(
        (-g*cmath.exp(-d*t) + 1)*nu**2)
    return psi_1, psi_2


@cfunc('double(double, CPointer(double))')
def _integrand(u, params):
    k, t, v, kappa, a, nu, rho = carray(params, (7,))
    psi_1, psi_2 = _heston_psi(u - 0.5j, t, kappa, a, nu, rho)
    return common._lipton_integrand(u, k, v, psi_1, psi_2)


def _reduced_formula(k, t, v, kappa, a, nu, rho, assert_no_arbitrage):
    strike = np.exp(k)
    no_arb_lb = np.maximum(0., 1. - strike)
    bs_c = blackscholes._reduced_formula(k, t, np.sqrt(v))
    if bs_c - no_arb_lb < 1e-5:
        return bs_c
    params = np.array([k, t, v, kappa, a, nu, rho]).ctypes.data_as(c_void_p)
    f = LowLevelCallable(_integrand.ctypes, params, 'double (double, void *)')
    c = 1 - quad(f, 0, np.inf)[0]/pi

    if assert_no_arbitrage:
        common._assert_no_arbitrage(1., c, strike)
    elif any(common._check_arbitrage(1, c, np.exp(k))):
        c = np.nan

    return c


@cfunc('double(double, CPointer(double))')
def _delta_integrand(u, params):
    k, t, v, kappa, a, nu, rho = carray(params, (7,))
    psi_1, psi_2 = _heston_psi(u - 1j, t, kappa, a, nu, rho)
    return common._delta_integrand(u, k, v, psi_1, psi_2)


@np.vectorize
def _reduced_delta(k, t, v, kappa, a, nu, rho):
    params = np.array([k, t, v, kappa, a, nu, rho]).ctypes.data_as(c_void_p)
    f = LowLevelCallable(_delta_integrand.ctypes, params,
                         'double (double, void *)')
    return 0.5 + quad(f, 0, np.inf)[0]/pi


@cfunc('double(double, CPointer(double))')
def _vega_integrand(u, params):
    k, t, v, kappa, a, nu, rho = carray(params, (7,))
    psi_1, psi_2 = _heston_psi(u - 0.5j, t, kappa, a, nu, rho)
    return common._vega_integrand(u, k, v, psi_1, psi_2)


@np.vectorize
def _reduced_vega(k, t, v, kappa, a, nu, rho):
    params = np.array([k, t, v, kappa, a, nu, rho]).ctypes.data_as(c_void_p)
    f = LowLevelCallable(_vega_integrand.ctypes, params,
                         'double (double, void *)')
    return -quad(f, 0, np.inf)[0]/pi


def _loss_xsect(cs, ks, ts, ws, params):
    v, kappa, a, nu, rho = params
    cs_heston = np.array([_reduced_formula(k, t, v, kappa, a, nu, rho, True)
                          for k, t in zip(ks, ts)])
    return ws*(cs_heston - cs)


def _reduced_calib_xsect(cs, ks, ts, ws, params):
    params, ier = leastsq(lambda params: _loss_xsect(cs, ks, ts, ws, params),
                          params)

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return params


def _loss_panel(cs, ks, ts, ws, params):
    vs = params[:-4]
    kappa, a, nu, rho = params[-4:]
    cs_heston = np.zeros(cs.shape)
    for i in range(len(vs)):
        cs_heston[i, :] = [
            _reduced_formula(k, t, vs[i], kappa, a, nu, rho, True)
            for k, t in zip(ks[i, :], ts)]
    return (ws*(cs_heston - cs)).flatten()


def _reduced_calib_panel(cs, ks, ts, ws, params):
    params, ier = leastsq(lambda params: _loss_panel(cs, ks, ts, ws, params),
                          params)

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return params


def _loss_vol(cs, ks, ts, ws, kappa, a, nu, rho, params):
    v, = params
    cs_heston = np.array(
        [_reduced_formula(k, t, v, kappa, a, nu, rho, True)
         for k, t in zip(ks, ts)])
    return ws*(cs_heston - cs)


def _reduced_calib_vol(cs, ks, ts, ws, kappa, a, nu, rho, params, n_cores):
    def loss(params):
        v, = params
        if n_cores is None:
            cs_heston = np.array(
                [_reduced_formula(k, t, v, kappa, a, nu, rho, True)
                 for k, t in zip(ks, ts)])
        else:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = executor.map(
                    _reduced_formula, ks, ts,
                    *map(repeat, (v, kappa, a, nu, rho, True))
                )
                cs_heston = np.array(list(futures))
        return ws * (cs_heston - cs)

    params, ier = leastsq(loss, params)

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return params
