import cmath
from ctypes import c_void_p
from math import pi

import numba as nb
import numpy as np
import scipy as sp
from numba import carray, cfunc, njit, objmode, types, typed
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import leastsq

from fyne import common
from fyne.wishart.rotation_count import count_rotations

COMPLEX_MATRIX_TYPE = types.complex128[:, :]


def formula(underlying_price, strike, expiry, vol, beta, q, m, r, put=False):
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
    k = np.log(strike/underlying_price)
    call = _reduced_formula(k, expiry, vol, beta, q, m, r) * underlying_price
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
    call_delta = _reduced_delta(k, expiry, vol, beta, q, m, r)
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
    return _reduced_vega(k, expiry, vol, beta, q, m, r) * underlying_price


@njit
def log_characteristic_function(u, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant, a):
    n = 2
    iu = 1j * u

    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    z[:n, :n] = m
    z[:n, -n:] = -2 * q.T @ q
    z[-n:, :n] = (iu * (iu  - 1) / 2) * np.eye(n)
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
