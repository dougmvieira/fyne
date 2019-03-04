import numpy as np
from scipy.optimize import newton
from scipy.stats import norm


def formula(underlying_price, strike, expiry, sigma):
    """Black-Scholes formula

    Computes the price of the option according to the Black-Scholes formula.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    sigma : float
        Volatility parameter.

    Returns
    -------
    float
        Option price according to Black-Scholes formula.

    Example
    -------

    >>> from fyne import blackscholes
    >>> sigma = 0.2
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> option_price = blackscholes.formula(underlying_price, strike, maturity,
    ...                                     sigma)
    >>> round(option_price, 2)
    11.77

    """

    k = np.log(strike/underlying_price)
    return _reduced_formula(k, expiry, sigma)*underlying_price


def vega(underlying_price, strike, expiry, sigma):
    """Black-Scholes Greek vega

    Computes the Greek vega -- i.e. the option price sensitivity with respect
    to its volatility parameter -- of the option price.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    sigma : float
        Volatility parameter.

    Returns
    -------
    float
        Greek vega according to Black-Scholes formula.

    Example
    -------

    >>> from fyne import blackscholes
    >>> sigma = 0.2
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> vega = blackscholes.vega(underlying_price, strike, maturity, sigma)
    >>> round(vega, 2)
    20.23

    """

    k = np.log(strike/underlying_price)
    return _reduced_vega(k, expiry, sigma)*underlying_price


def implied_vol(underlying_price, strike, expiry, option_price,
                initial_guess=0.5):
    """Implied volatility function

    Inverts the Black-Scholes formula to find the volatility that matches the
    given option price. The implied volatility is computed using Newton's
    method.

    Parameters
    ----------
    underlying_price : float
        Price of the underlying asset.
    strike : float
        Strike of the option.
    expiry : float
        Time remaining until the expiry of the option.
    option_price : float
        Option price according to Black-Scholes formula.
    initial_guess : float, optional
        Initial guess for the implied volatility for the Newton's method.

    Returns
    -------
    float
        Implied volatility.

    Example
    -------

    >>> from fyne import blackscholes
    >>> option_price = 11.77
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> implied_vol = blackscholes.implied_vol(underlying_price, strike,
    ...                                        maturity, option_price)
    >>> round(implied_vol, 2)
    0.2

    """
    if (option_price <= underlying_price - strike
            or option_price >= underlying_price):
        raise ValueError("Option price outside no-arbitrage bounds")

    k = np.log(strike/underlying_price)
    c = option_price/underlying_price
    return _reduced_implied_vol(k, expiry, c, initial_guess)


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
