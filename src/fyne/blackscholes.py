import numpy as np
from scipy.optimize import newton
from scipy.stats import norm


def formula(underlying_price, strike, maturity, sigma):
    """Black-Scholes formula
    """

    k = np.log(strike/underlying_price)
    return _reduced_formula(k, maturity, sigma)*underlying_price


def vega(underlying_price, strike, maturity, sigma):
    """Black-Scholes Greek vega
    """

    k = np.log(strike/underlying_price)
    return _reduced_vega(k, maturity, sigma)*underlying_price


def implied_vol(underlying_price, strike, maturity, option_price,
                initial_guess=0.5):
    """Implied volatility function
    """
    if (option_price <= underlying_price - strike
            or option_price >= underlying_price):
        raise ValueError("Option price outside no-arbitrage bounds")

    k = np.log(strike/underlying_price)
    c = option_price/underlying_price
    return _reduced_implied_vol(k, maturity, c, initial_guess)


def _reduced_formula(k, t, sigma):
    """Reduced Black-Scholes formula

    Used in `fyne.blackscholes.formula`.
    """
    tot_std = sigma*np.sqrt(t)
    d_plus = tot_std/2 - k/tot_std
    d_minus = d_plus - tot_std

    return norm.cdf(d_plus) - norm.cdf(d_minus)*np.exp(k)


def _reduced_vega(k, t, sigma):
    """Reduced Black-Scholes Greek vega

    Used in `fyne.blackscholes.vega`.
    """
    tot_std = sigma*np.sqrt(t)
    d_plus = tot_std/2 - k/tot_std

    return norm.pdf(d_plus)*np.sqrt(t)


def _reduced_implied_vol(k, t, c, iv0):
    """Reduced Implied volatility function

    Used in `fyne.blackscholes.implied_vol`.
    """

    return newton(lambda iv: _reduced_formula(k, t, iv) - c, iv0,
                  lambda iv: _reduced_vega(k, t, iv))
