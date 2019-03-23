import cmath

import numpy as np
from numba import njit


def _assert_no_arbitrage(underlying_price, option_price, strike):
    no_arb_low_bound = np.maximum(0., underlying_price - strike)
    no_arb_upper_bound = underlying_price

    if np.any(option_price <= no_arb_low_bound):
        raise ValueError("Warning: Option price below no-arbitrage bounds")
    elif np.any(option_price >= no_arb_upper_bound):
        raise ValueError("Warning: Option price above no-arbitrage bounds")


def _put_call_parity(call, underlying_price, strike, put_bool):
    return np.where(put_bool, call - underlying_price + strike, call)


def _put_call_parity_reverse(put, underlying_price, strike, put_bool):
    return np.where(put_bool, put + underlying_price - strike, put)


def _put_call_parity_delta(call_delta, put_bool):
    return np.where(put_bool, call_delta - 1., call_delta)


@njit
def _lipton_integrand(u, k, v, psi_1, psi_2):
    return cmath.exp((0.5 - u*1j)*k + psi_1 + psi_2*v).real/(u**2 + 0.25)


@njit
def _delta_integrand(u, k, v, psi_1, psi_2):
    return (cmath.exp(-u*k*1j + psi_1 + v*psi_2)/(u*1j)).real


@njit
def _vega_integrand(u, k, v, psi_1, psi_2):
    return (psi_2*cmath.exp((0.5 - u*1j)*k + psi_1 + psi_2*v)
            ).real/(u**2 + 0.25)
