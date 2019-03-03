import numpy as np
from scipy.optimize import newton
from scipy.stats import norm


def formula(k, t, sigma):
    """Black-Scholes formula
    """
    tot_std = sigma*np.sqrt(t)
    d_plus = tot_std/2 - k/tot_std
    d_minus = d_plus - tot_std

    return norm.cdf(d_plus) - norm.cdf(d_minus)*np.exp(k)


def vega(k, t, sigma):
    """Black-Scholes Greek vega
    """
    tot_std = sigma*np.sqrt(t)
    d_plus = tot_std/2 - k/tot_std

    return norm.pdf(d_plus)*np.sqrt(t)


def implied_vol(k, t, c, iv0=0.5):
    """Implied volatility function
    """
    if c <= 0 or c >= 1:
        raise ValueError("Option price outside no-arbitrage bounds")

    return newton(lambda iv: formula(k, t, iv) - c, iv0,
                  lambda iv: vega(k, t, iv))
