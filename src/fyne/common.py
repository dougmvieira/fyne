import cmath

from numba import njit


@njit
def _lipton_integrand(u, k, v, psi_1, psi_2):
    return cmath.exp((0.5 - u*1j)*k + psi_1 + psi_2*v).real/(u**2 + 0.25)
