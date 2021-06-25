from math import erf, exp, pi, sqrt

import numba as nb
import numpy as np
from numba import float64, int64
from py_lets_be_rational import lets_be_rational

from fyne import common


def formula(underlying_price, strike, expiry, sigma, put=False):
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
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.

    Returns
    -------
    float
        Option price according to Black-Scholes formula.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import blackscholes
    >>> sigma = 0.2
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> expiry = 0.5
    >>> call_price = blackscholes.formula(underlying_price, strike, expiry,
    ...                                   sigma)
    >>> np.round(call_price, 2)
    11.77
    >>> put_price = blackscholes.formula(underlying_price, strike, expiry,
    ...                                  sigma, put=True)
    >>> np.round(put_price, 2)
    1.77

    """
    k = np.array(np.log(strike/underlying_price))
    expiry, sigma = map(np.array, (expiry, sigma))
    call = _reduced_formula(k, expiry, sigma)*underlying_price
    return common._put_call_parity(call, underlying_price, strike, put)


def implied_vol(underlying_price, strike, expiry, option_price, put=False,
                assert_no_arbitrage=True):
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
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.
    assert_no_arbitrage : bool, optional
        Whether to throw an exception upon no arbitrage bounds violation.
        Defaults to `True`.

    Returns
    -------
    float
        Implied volatility.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import blackscholes
    >>> call_price = 11.77
    >>> put_price = 1.77
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> expiry = 0.5
    >>> implied_vol = blackscholes.implied_vol(underlying_price, strike,
    ...                                        expiry, call_price)
    >>> np.round(implied_vol, 2)
    0.2
    >>> implied_vol = blackscholes.implied_vol(underlying_price, strike,
    ...                                        expiry, put_price, put=True)
    >>> np.round(implied_vol, 2)
    0.2

    """
    call = common._put_call_parity_reverse(option_price, underlying_price,
                                           strike, put)
    b = option_price / np.sqrt(underlying_price * strike)
    x = np.log(underlying_price / strike)
    theta = np.where(put, -1, 1)

    if assert_no_arbitrage:
        common._assert_no_arbitrage(underlying_price, call, strike)

    call, b, x, theta, expiry = np.broadcast_arrays(call, b, x, theta, expiry)
    noarb_mask = ~np.any(
        common._check_arbitrage(underlying_price, call, strike), axis=0)
    noarb_mask &= ~np.any(
        tuple(map(np.isnan, (b, x, theta, expiry))), axis=0)

    iv = np.full(call.shape, np.nan)
    iv[noarb_mask] = _reduced_implied_vol(
        b[noarb_mask], x[noarb_mask], theta[noarb_mask], 2
    ) / np.sqrt(expiry[noarb_mask])
    return iv


def delta(underlying_price, strike, expiry, sigma, put=False):
    """Black-Scholes Greek delta

    Computes the Greek delta of the option -- i.e. the option price sensitivity
    with respect to its underlying price -- according to the Black-Scholes
    model.

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
    put : bool, optional
        Whether the option is a put option. Defaults to `False`.

    Returns
    -------
    float
        Greek delta according to Black-Scholes formula.

    Example
    -------

    >>> import numpy as np
    >>> from fyne import blackscholes
    >>> sigma = 0.2
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> expiry = 0.5
    >>> call_delta = blackscholes.delta(underlying_price, strike, expiry, sigma)
    >>> np.round(call_delta, 2)
    0.79
    >>> put_delta = blackscholes.delta(underlying_price, strike, expiry, sigma,
    ...                                put=True)
    >>> np.round(put_delta, 2)
    -0.21

    """
    k = np.array(np.log(strike/underlying_price))
    expiry, sigma = map(np.array, (expiry, sigma))
    call_delta = _reduced_delta(k, expiry, sigma)
    return common._put_call_parity_delta(call_delta, put)


def vega(underlying_price, strike, expiry, sigma):
    """Black-Scholes Greek vega

    Computes the Greek vega of the option -- i.e. the option price sensitivity
    with respect to its volatility parameter -- according to the Black-Scholes
    model. Note that the Greek vega is the same for calls and puts.

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

    >>> import numpy as np
    >>> from fyne import blackscholes
    >>> sigma = 0.2
    >>> underlying_price = 100.
    >>> strike = 90.
    >>> maturity = 0.5
    >>> vega = blackscholes.vega(underlying_price, strike, maturity, sigma)
    >>> np.round(vega, 2)
    20.23

    """
    k = np.array(np.log(strike/underlying_price))
    expiry, sigma = map(np.array, (expiry, sigma))
    return _reduced_vega(k, expiry, sigma)*underlying_price


@nb.vectorize([float64(float64)])
def _norm_cdf(z):
    return (1 + erf(z/sqrt(2)))/2


@nb.vectorize([float64(float64)])
def _norm_pdf(z):
    return exp(-z**2/2)/sqrt(2*pi)


@nb.vectorize([float64(float64, float64, float64)], nopython=True)
def _reduced_formula(k, t, sigma):
    """Reduced Black-Scholes formula

    Used in `fyne.blackscholes.formula`.
    """
    tot_std = sigma*sqrt(t)
    d_plus = tot_std/2 - k/tot_std
    d_minus = d_plus - tot_std

    return _norm_cdf(d_plus) - _norm_cdf(d_minus)*exp(k)


@nb.vectorize([float64(float64, float64, float64)], nopython=True)
def _reduced_vega(k, t, sigma):
    """Reduced Black-Scholes Greek vega

    Used in `fyne.blackscholes.vega`.
    """
    tot_std = sigma*sqrt(t)
    d_plus = tot_std/2 - k/tot_std

    return _norm_pdf(d_plus)*sqrt(t)


_reduced_implied_vol = nb.vectorize(
    [float64(float64, float64, float64, int64)], nopython=True)(
    vars(lets_be_rational)[
        '_unchecked_normalised_implied_volatility_from_a_transformed_rational_'
        'guess_with_limited_iterations'])


@nb.vectorize([float64(float64, float64, float64)], nopython=True)
def _reduced_delta(k, t, sigma):
    """Reduced Black-Scholes Greek delta

    Used in `fyne.blackscholes.delta`.
    """
    tot_std = sigma*sqrt(t)
    d_plus = tot_std/2 - k/tot_std

    return _norm_cdf(d_plus)
