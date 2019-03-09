import cmath
from itertools import repeat
from math import pi
from timeit import Timer

import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.optimize import leastsq

from fyne import common


def formula(underlying_price, strike, expiry, vol, kappa, theta, nu, rho):
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

    Returns
    -------
    float
        Option price according to Heston formula.

    Example
    -------

    >>> from fyne import heston
    >>> v, kappa, theta, nu, rho = 0.2, 1.3, 0.04, 0.4, -0.3
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> option_price = heston.formula(underlying_price, strike, maturity,
    ...                               v, kappa, theta, nu, rho)
    >>> round(option_price, 2)
    16.32

    """

    k = np.log(strike/underlying_price)
    a = kappa*theta
    return _reduced_formula(k, expiry, vol, kappa, a, nu, rho)*underlying_price


def delta(underlying_price, strike, expiry, vol, kappa, theta, nu, rho):
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

    Returns
    -------
    float
        Option Greek :math:`\Delta` (delta) according to Heston formula.

    Example
    -------

    >>> from fyne import heston
    >>> v, kappa, theta, nu, rho = 0.2, 1.3, 0.04, 0.4, -0.3
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> delta = heston.delta(underlying_price, strike, maturity, v, kappa,
    ...                      theta, nu, rho)
    >>> round(delta, 2)
    0.72

    """

    k = np.log(strike/underlying_price)
    a = kappa*theta
    return _reduced_delta(k, expiry, vol, kappa, a, nu, rho)


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

    >>> from fyne import heston
    >>> v, kappa, theta, nu, rho = 0.2, 1.3, 0.04, 0.4, -0.3
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> vega = heston.vega(underlying_price, strike, maturity, v, kappa, theta,
    ...                    nu, rho)
    >>> round(vega, 2)
    22.5

    """

    k = np.log(strike/underlying_price)
    a = kappa*theta
    return _reduced_vega(k, expiry, vol, kappa, a, nu, rho)*underlying_price


def calibration_crosssectional(underlying_price, strikes, expiries,
                               option_prices, initial_guess):
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
    initial_guess : float, optional
        Initial guess for instantaneous volatility :math:`V_0` and the Heston
        parameters :math:`\kappa`, :math:`\theta`, :math:`\nu` and
        :math:`\rho`, respectively.

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
    >>> option_prices = np.array(
    ...     [heston.formula(underlying_price, strike, expiry, vol, kappa,
    ...                     theta, nu, rho)
    ...      for strike, expiry in zip(strikes, expiries)])
    >>> initial_guess = np.array([vol + 0.01, kappa + 1, theta + 0.01,
    ...                           nu - 0.1, rho - 0.1])
    >>> calibrated = heston.calibration_crosssectional(
    ...     underlying_price, strikes, expiries, option_prices, initial_guess)
    >>> [round(param, 4) for param in calibrated]
    [0.0457, 5.07, 0.0457, 0.48, -0.767]

    """

    cs = option_prices/underlying_price
    ks = np.log(strikes/underlying_price)
    vol0, kappa0, theta0, nu0, rho0 = initial_guess
    params = np.array([vol0, kappa0, kappa0*theta0, nu0, rho0])

    vol, kappa, a, nu, rho = _reduced_calib_xsect(cs, ks, expiries, params)

    return vol, kappa, a/kappa, nu, rho


def calibration_panel(underlying_prices, strikes, expiries, option_prices,
                      initial_guess):
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
    initial_guess : float, optional
        Initial guess for instantaneous volatility :math:`V_0` and the Heston
        parameters :math:`\kappa`, :math:`\theta`, :math:`\nu` and
        :math:`\rho`, respectively.

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
    >>> option_prices = np.zeros((len(underlying_prices), len(strikes)))
    >>> for i in range(len(underlying_prices)):
    ...     option_prices[i, :] = [
    ...         heston.formula(underlying_prices[i], strike, expiry, vols[i],
    ...                        kappa, theta, nu, rho)
    ...         for strike, expiry in zip(strikes, expiries)]
    >>> initial_guess = np.array([vols[1] + 0.01, kappa + 1, theta + 0.01,
    ...                           nu - 0.1, rho - 0.1])
    >>> vols, kappa, theta, nu, rho = heston.calibration_panel(
    ...     underlying_prices, strikes, expiries, option_prices, initial_guess)
    >>> np.round(vols, 4)
    array([0.05 , 0.045, 0.055])
    >>> [round(param, 4) for param in (kappa, theta, nu, rho)]
    [5.07, 0.0457, 0.48, -0.767]

    """

    cs = option_prices/underlying_prices[:, None]
    ks = np.log(strikes[None, :]/underlying_prices[:, None])
    vol0, kappa0, theta0, nu0, rho0 = initial_guess
    params = vol0*np.ones(len(underlying_prices) + 4)
    params[-4:] = kappa0, kappa0*theta0, nu0, rho0

    calibrated = _reduced_calib_panel(cs, ks, expiries, params)
    vols = calibrated[:-4]
    kappa, a, nu, rho = calibrated[-4:]

    return vols, kappa, a/kappa, nu, rho


def calibration_vol(underlying_price, strikes, expiries, option_prices, kappa,
                    theta, nu, rho, vol_guess=0.1):
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
    vol_guess : float, optional
        Initial guess for instantaneous volatility :math:`V_0`. Defaults to
        0.1.

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
    >>> option_prices = np.array(
    ...     [heston.formula(underlying_price, strike, expiry, vol, kappa,
    ...                     theta, nu, rho)
    ...      for strike, expiry in zip(strikes, expiries)])
    >>> calibrated_vol = heston.calibration_vol(
    ...     underlying_price, strikes, expiries, option_prices, kappa, theta,
    ...     nu, rho)
    >>> round(calibrated_vol, 4)
    0.0457

    """

    cs = option_prices/underlying_price
    ks = np.log(strikes/underlying_price)

    vol, = _reduced_calib_vol(cs, ks, expiries, kappa, kappa*theta, nu, rho,
                              np.array([vol_guess]))
    return vol


def benchmark(n):
    """Benchmarking function for the Heston formula implementation

    This function computes the time elapsed to evaluate the Heston formula
    under some example parameters for a specific maturity and a given number of
    strikes. The bounds of the strikes is kept fixed.

    Parameters
    ----------
    n : int
        Number of strikes to be evaluated

    Returns
    -------
    tuple of float
        Returns the time spent performing the Heston formula computations and
    its standard deviation.

    """

    setup = """
import numpy as np

from fyne.heston import _reduced_formula


t, v, kappa, a, nu, rho = 0.5, 0.0457, 5.07, 0.2317, 0.48, -0.767
ks = np.linspace(np.log(0.8), np.log(1.2), {})

# First execution to trigger JIT
_reduced_formula(0., t, v, kappa, a, nu, rho)""".format(n)

    timer = Timer('[_reduced_formula(k, t, v, kappa, a, nu, rho) for k in ks]',
                  setup=setup)
    return min(map(lambda t: t.timeit(number=50)/50., repeat(timer, 5)))


@njit
def _heston_psi(u, t, kappa, a, nu, rho):
    d = cmath.sqrt(nu**2*(u**2 + 1j*u) + (kappa - 1j*nu*rho*u)**2)
    g = (-d + kappa - 1j*nu*rho*u)/(d + kappa - 1j*nu*rho*u)
    h = (g*cmath.exp(-d*t) - 1)/(g - 1)
    psi_1 = a*(t*(-d + kappa - 1j*nu*rho*u) - 2*cmath.log(h))/nu**2
    psi_2 = (1 - cmath.exp(-d*t))*(-d + kappa - 1j*nu*rho*u)/(
        (-g*cmath.exp(-d*t) + 1)*nu**2)
    return psi_1, psi_2


@njit
def _integrand(u, k, t, v, kappa, a, nu, rho):
    psi_1, psi_2 = _heston_psi(u - 0.5j, t, kappa, a, nu, rho)
    return common._lipton_integrand(u, k, v, psi_1, psi_2)


def _reduced_formula(k, t, v, kappa, a, nu, rho):
    c = 1 - quad(lambda u: _integrand(u, k, t, v, kappa, a, nu, rho), 0,
                 np.inf)[0]/pi

    common._assert_no_arbitrage(1., c, np.exp(k))

    return c


@njit
def _delta_integrand(u, k, t, v, kappa, a, nu, rho):
    psi_1, psi_2 = _heston_psi(u - 1j, t, kappa, a, nu, rho)
    return common._delta_integrand(u, k, v, psi_1, psi_2)


def _reduced_delta(k, t, v, kappa, a, nu, rho):
    return 0.5 + quad(lambda u: _delta_integrand(u, k, t, v, kappa, a, nu,
                                                 rho), 0, np.inf)[0]/pi


@njit
def _vega_integrand(u, k, t, v, kappa, a, nu, rho):
    psi_1, psi_2 = _heston_psi(u - 0.5j, t, kappa, a, nu, rho)
    return common._vega_integrand(u, k, v, psi_1, psi_2)


def _reduced_vega(k, t, v, kappa, a, nu, rho):
    return -quad(lambda u: _vega_integrand(u, k, t, v, kappa, a, nu, rho), 0,
                 np.inf)[0]/pi


def _loss_xsect(cs, ks, ts, params):
    v, kappa, a, nu, rho = params
    cs_heston = np.array([_reduced_formula(k, t, v, kappa, a, nu, rho)
                          for k, t in zip(ks, ts)])
    return cs_heston - cs


def _reduced_calib_xsect(cs, ks, ts, params):
    params, ier = leastsq(lambda params: _loss_xsect(cs, ks, ts, params),
                          params)

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return params


def _loss_panel(cs, ks, ts, params):
    vs = params[:-4]
    kappa, a, nu, rho = params[-4:]
    cs_heston = np.zeros(cs.shape)
    for i in range(len(vs)):
        cs_heston[i, :] = [_reduced_formula(k, t, vs[i], kappa, a, nu, rho)
                           for k, t in zip(ks[i, :], ts)]
    return (cs_heston - cs).flatten()


def _reduced_calib_panel(cs, ks, ts, params):
    params, ier = leastsq(lambda params: _loss_panel(cs, ks, ts, params),
                          params)

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return params


def _loss_vol(cs, ks, ts, kappa, a, nu, rho, params):
    v, = params
    cs_heston = np.array([_reduced_formula(k, t, v, kappa, a, nu, rho)
                          for k, t in zip(ks, ts)])
    return cs_heston - cs


def _reduced_calib_vol(cs, ks, ts, kappa, a, nu, rho, params):
    params, ier = leastsq(lambda params: _loss_vol(cs, ks, ts, kappa, a, nu,
                                                   rho, params), params)

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return params
