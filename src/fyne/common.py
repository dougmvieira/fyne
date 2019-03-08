import cmath

from numba import njit


def _assert_no_arbitrage(underlying_price, option_price, strike):
    no_arb_low_bound = max(0., underlying_price - strike)
    no_arb_upper_bound = underlying_price

    if option_price <= no_arb_low_bound:
        raise ValueError("Warning: Option price below no-arbitrage bounds")
    elif option_price >= no_arb_upper_bound:
        raise ValueError("Warning: Option price above no-arbitrage bounds")


@njit
def _delta_integrand(u, k, v, psi_1, psi_2):
    return (cmath.exp(-u*k*1j + psi_1 + v*psi_2)/(u*1j)).real


@njit
def _lipton_integrand(u, k, v, psi_1, psi_2):
    return cmath.exp((0.5 - u*1j)*k + psi_1 + psi_2*v).real/(u**2 + 0.25)
