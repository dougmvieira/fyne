import cmath
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numba as nb
import numpy as np
from numba import njit, objmode, types, typed
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import leastsq

from fyne import common
from fyne.wishart.rotation_count import count_rotations
from fyne.wishart.utils import pack_params, pack_vol, unpack_params, unpack_vol

COMPLEX_MATRIX_TYPE = types.complex128[:, :]


def formula(underlying_price, strike, expiry, vol, beta, q, m, r, put=False,
            n_cores=None):
    r"""Wishart model formula

    Computes the price of the option according to the Wishart model formula.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    vol : float matrix
        Instantaneous volatility matrix.
    beta: float
        Model parameter :math:`\beta`.
    q : float matrix
        Model parameter :math:`Q`.
    m : float matrix
        Model parameter :math:`M`.
    r : float matrix
        Model parameter :math:`R`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.

    Returns
    -------
    float
        Option price according to Wishart model formula.
    """

    if vol.shape != (2, 2):
        raise NotImplementedError('Only 2-factor Wishart is currently available.')
    ks = np.log(strike/underlying_price)
    broadcasted = np.broadcast(ks, expiry)
    call = np.empty(broadcasted.shape)
    if n_cores is None:
        call.flat = [_reduced_formula(k, t, vol, beta, q, m, r)
                     for (k, t) in broadcasted]
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            call.flat = list(executor.map(_reduced_formula, *zip(*broadcasted), *map(repeat, (vol, beta, q, m, r))))
    call *= underlying_price
    return common._put_call_parity(call, underlying_price, strike, put)


def delta(underlying_price, strike, expiry, vol, beta, q, m, r, put=False):
    r"""Wishart Greek delta

    Computes the Greek :math:`\Delta` (delta) of the option according to the
    Wishart model.

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
    beta: float
        Model parameter :math:`\beta`.
    q : float matrix
        Model parameter :math:`Q`.
    m : float matrix
        Model parameter :math:`M`.
    r : float matrix
        Model parameter :math:`R`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.

    Returns
    -------
    float
        Option Greek :math:`\Delta` (delta) according to Wishart formula.

    """

    k = np.log(strike/underlying_price)

    @np.vectorize
    def vec_reduced_delta(k, t):
        return _reduced_delta(k, t, vol, beta, q, m, r)
    call_delta = vec_reduced_delta(k, expiry)
    return common._put_call_parity_delta(call_delta, put)


def vega(underlying_price, strike, expiry, vol, beta, q, m, r):
    r"""Wishart Greek vega

    Computes the Greek :math:`\mathcal{V}` (vega) of the option according to
    the Wishart model. This is the gradient of the option price with respect to
    the volatility matrix.

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
    beta: float
        Model parameter :math:`\beta`.
    q : float matrix
        Model parameter :math:`Q`.
    m : float matrix
        Model parameter :math:`M`.
    r : float matrix
        Model parameter :math:`R`.

    Returns
    -------
    float
        Option Greek :math:`\mathcal{V}` (vega) according to Wishart formula.

    """

    k = np.log(strike/underlying_price)

    @np.vectorize
    def vec_reduced_vega(k, t):
        return _reduced_vega(k, t, vol, beta, q, m, r)
    return vec_reduced_vega(k, expiry) * underlying_price


def calibration_crosssectional(underlying_price, strikes, expiries,
                               option_prices, initial_guess, put=False,
                               weights=None, n_cores=None):
    r"""Wishart cross-sectional calibration

    Recovers the Wishart model parameters from options prices at a single point
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
        Initial guess for instantaneous volatility matrix :math:`\Sigma_0` and
        the Wishart parameters :math:`\beta`, :math:`Q`, :math:`M` and
        :math:`R`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.
    weights : numpy.array, optional
        One-dimensional array of call options prices. Must be of the same
        length as the option_prices, expiries and strikes arrays.

    Returns
    -------
    tuple
        Returns the calibrated instantaneous volatility :math:`V_0` and the
        Wishart parameters :math:`\kappa`, :math:`\theta`, :math:`\nu` and
        :math:`\rho`, respectively, as :obj:`float`.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import wishart
    >>> params = dict(
    >>>     vol=np.array([[0.0327, 0.0069],
    >>>                   [0.0069, 0.0089]]),
    >>>     beta=0.6229,
    >>>     q=np.array([[0.3193, 0.2590],
    >>>                 [0.2899, 0.2469]]),
    >>>     m=np.array([[-0.9858, -0.5224],
    >>>                 [-0.1288, -0.9746]]),
    >>>     r=np.array([[-0.2116, -0.4428],
    >>>                 [-0.2113, -0.5921]]),
    >>> )
    >>> underlying_price = 1640.
    >>> strikes = np.array([1148., 1148., 1148., 1148.,
    ...                     1312., 1312., 1312., 1312.,
    ...                     1640., 1640., 1640., 1640.,
    ...                     1968., 1968., 1968., 1968.,
    ...                     2296., 2296., 2296., 2296.])
    >>> expiries = np.array([0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5])
    >>> put = np.array([False, False, False, False,
    ...                 False, False, False, False,
    ...                 False, False, False, False,
    ...                 False, False, False, False,
    ...                 True, True, True, True])
    >>> option_prices = wishart.formula(underlying_price, strikes, expiries,
    ...                                 put=put, **params)
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
    cs = calls / underlying_price
    ks = np.log(strikes/underlying_price)
    ws = 1 / cs if weights is None else weights / cs

    vol0, beta0, q0, m0, r0 = initial_guess
    if vol0.shape != (2, 2):
        raise NotImplementedError('Only 2-factor Wishart is currently available.')

    return _reduced_calib_xsect(cs, ks, expiries, ws, vol0, beta0, q0, m0, r0, n_cores)


def calibration_vol(underlying_price, strikes, expiries, option_prices,
                    vol_guess, beta, q, m, r, put=False, weights=None,
                    n_cores=None):
    r"""Wishart volatility matrix calibration

    Recovers the Wishart instantaneous volatility matrix from options prices at
    a single point in time. The Wishart model parameters must be provided. The
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
    vol_guess : float, optional
        Initial guess for instantaneous volatility :math:`V_0`. Defaults to
        0.1.
    beta: float
        Model parameter :math:`\beta`.
    q : float matrix
        Model parameter :math:`Q`.
    m : float matrix
        Model parameter :math:`M`.
    r : float matrix
        Model parameter :math:`R`.
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.
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
    >>> from fyne import wishart
    >>> params = dict(
    >>>     vol=np.array([[0.0327, 0.0069],
    >>>                   [0.0069, 0.0089]]),
    >>>     beta=0.6229,
    >>>     q=np.array([[0.3193, 0.2590],
    >>>                 [0.2899, 0.2469]]),
    >>>     m=np.array([[-0.9858, -0.5224],
    >>>                 [-0.1288, -0.9746]]),
    >>>     r=np.array([[-0.2116, -0.4428],
    >>>                 [-0.2113, -0.5921]]),
    >>> )
    >>> underlying_price = 1640.
    >>> strikes = np.array([1148., 1148., 1148., 1148.,
    ...                     1312., 1312., 1312., 1312.,
    ...                     1640., 1640., 1640., 1640.,
    ...                     1968., 1968., 1968., 1968.,
    ...                     2296., 2296., 2296., 2296.])
    >>> expiries = np.array([0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5,
    ...                      0.12, 0.19, 0.25, 0.5])
    >>> put = np.array([False, False, False, False,
    ...                 False, False, False, False,
    ...                 False, False, False, False,
    ...                 False, False, False, False,
    ...                 True, True, True, True])
    >>> option_prices = wishart.formula(underlying_price, strikes, expiries,
    ...                                 put=put, **params)
    >>> calibrated_vol = wishart.calibration_vol(
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

    return _reduced_calib_vol(cs, ks, expiries, ws, vol_guess, beta, q, m, r,
                              n_cores)


@njit(
    types.complex128(
        types.complex128, types.float64, types.float64[:, :], types.float64,
        types.float64[:, :], types.float64[:, :], types.float64[:, :],
        types.ListType(types.complex128), types.complex128[:], types.int64[:],
        types.complex128[:, :]))
def log_characteristic_function(u, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant, a):
    n = 2
    iu = 1j * u

    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    z[:n, :n] = m
    z[:n, -n:] = -2 * q.T @ q
    z[-n:, :n] = (iu * (iu - 1) / 2) * np.eye(n)
    z[-n:, -n:] = -(m.T + 2 * iu * (r.T @ q))

    with objmode(exp_tz=COMPLEX_MATRIX_TYPE):
        exp_tz = expm(t * z)
    if np.any(np.isnan(exp_tz)) or np.any(np.isinf(exp_tz)):
        return -np.inf
    if np.absolute(np.trace(exp_tz)) > 1e9:
        return -np.inf
    g = exp_tz[-n:, :n]
    f = exp_tz[-n:, -n:]

    rot = count_rotations(u, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant)
    log_det_f = cmath.log(np.linalg.det(f)) + 2j * cmath.pi * rot

    a[:, :] = np.linalg.solve(f, g)
    c = (
        -iu * t * beta * np.sum(r * q)
        - (beta / 2) * (np.trace(m) * t + log_det_f)
    )
    return np.sum(a * v) + c


def _reduced_formula(k, t, v, beta, q, m, r):
    rot_locs = nb.typed.List(lsttype=types.ListType(types.complex128))
    cached_u, cached_quadrant = np.array([-0.5j]), np.array([0])
    a = np.zeros((2, 2), dtype=np.complex128)

    def integrand(u):
        psi = log_characteristic_function(u - 0.5j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant, a)
        integrand = cmath.exp(psi + (0.5 - 1j * u) * k).real / (u ** 2 + 0.25)
        return integrand
    return 1 - quad(integrand, 0, np.inf)[0] / np.pi


def _reduced_delta(k, t, v, beta, q, m, r):
    rot_locs = typed.List(lsttype=types.ListType(types.complex128))
    cached_u, cached_quadrant = np.array([-1j]), np.array([0])
    a = np.zeros((2, 2), dtype=np.complex128)
    psi_dem = log_characteristic_function(-1j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant, a)

    def integrand(u):
        psi = log_characteristic_function(u - 1j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant, a)
        integrand = (cmath.exp(-1j * u * k + psi - psi_dem) / (1j * u)).real
        return integrand
    return 0.5 + quad(integrand, 0, np.inf)[0] / np.pi


def _reduced_vega(k, t, v, beta, q, m, r):
    n = 2
    rot_locs = nb.typed.List(lsttype=types.ListType(types.complex128))
    cached_u, cached_quadrant = np.array([-0.5j]), np.array([0])
    a = np.zeros((n, n), dtype=np.complex128)
    vega = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            def integrand(u):
                psi = log_characteristic_function(u - 0.5j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant, a)
                integrand = (a[i, j] * cmath.exp(psi + (0.5 - 1j * u) * k)).real / (u ** 2 + 0.25)
                return integrand
            vega[i, j] = - quad(integrand, 0, np.inf)[0] / np.pi
    return vega


def _reduced_calib_xsect(cs, ks, ts, ws, v, beta, q, m, r, n_cores):
    n = 2

    def loss(params):
        v, beta, q, m, r = unpack_params(params, n)
        print(f'v = {v}')
        print(f'beta = {beta}')
        print(f'q = {q}')
        print(f'm = {m}')
        print(f'r = {r}')
        print()
        if n_cores is None:
            cs_wishart = np.array([_reduced_formula(k, t, v, beta, q, m, r)
                                   for k, t in zip(ks, ts)])
        else:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                cs_wishart = np.array(list(executor.map(_reduced_formula, ks, ts, *map(repeat, (v, beta, q, m, r)))))
        return ws * (cs_wishart - cs)

    params, ier = leastsq(loss, pack_params(v, beta, q, m, r, n))

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Wishart calibration failed. ier = {}".format(ier))

    return unpack_params(params, n)


def _reduced_calib_vol(cs, ks, ts, ws, params, beta, q, m, r, n_cores):
    n = 2

    def loss(params):
        v = unpack_vol(params, n)
        print(f'v = {v}')
        if n_cores is None:
            cs_wishart = np.array([_reduced_formula(k, t, v, beta, q, m, r)
                                   for k, t in zip(ks, ts)])
        else:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                cs_wishart = np.array(list(executor.map(_reduced_formula, ks, ts, *map(repeat, (v, beta, q, m, r)))))
        return ws * (cs_wishart - cs)

    params, ier = leastsq(loss, pack_vol(params, n))

    if ier not in [1, 2, 3, 4]:
        raise ValueError("Heston calibration failed. ier = {}".format(ier))

    return unpack_vol(params, n)
