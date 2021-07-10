import cmath
from ctypes import c_void_p
from math import pi

import numpy as np
from numba import carray, cfunc, njit, types
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.optimize import leastsq

from fyne import common


def formula(underlying_price, strike, expiry, vol, kappa, theta, nu, rho,
            put=False):
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
        Option price according to Wishart model formula.
    """

    k = np.log(strike/underlying_price)
    a = kappa*theta
    call = _reduced_formula(k, expiry, vol, kappa, a, nu, rho)*underlying_price
    return common._put_call_parity(call, underlying_price, strike, put)


@njit
def _wishart_psi(u, t, beta, q, m, r):
    d = cmath.sqrt(nu**2*(u**2 + 1j*u) + (kappa - 1j*nu*rho*u)**2)
    g = (-d + kappa - 1j*nu*rho*u)/(d + kappa - 1j*nu*rho*u)
    h = (g*cmath.exp(-d*t) - 1)/(g - 1)
    psi_1 = a*(t*(-d + kappa - 1j*nu*rho*u) - 2*cmath.log(h))/nu**2
    psi_2 = (1 - cmath.exp(-d*t))*(-d + kappa - 1j*nu*rho*u)/(
        (-g*cmath.exp(-d*t) + 1)*nu**2)
    return psi_1, psi_2


@cfunc('double(double, CPointer(double))')
def _integrand_2factor(u, params):
    params_arr = carray(params, (16,))
    beta = params_arr[0]
    v = np.empty((2, 2))
    v[:, 0] = params_arr[1:3]
    v[:, 1] = params_arr[2:4]
    q = np.reshape(params_arr[4:8], (2, 2))
    m = np.reshape(params_arr[8:12], (2, 2))
    r = np.reshape(params_arr[12:16], (2, 2))
    psi_1, psi_2 = _wishart_psi(u - 0.5j, t, beta, q, m, r)
    return common._lipton_integrand(u, k, v, psi_1, psi_2)


@np.vectorize
def _reduced_formula(k, t, v, kappa, a, nu, rho):
    params = np.array([k, t, v, kappa, a, nu, rho]).ctypes.data_as(c_void_p)
    f = LowLevelCallable(_integrand.ctypes, params, 'double (double, void *)')
    c = 1 - quad(f, 0, np.inf)[0]/pi

    common._assert_no_arbitrage(1., c, np.exp(k))

    return c
