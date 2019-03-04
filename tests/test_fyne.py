from pytest import raises
from fyne import blackscholes


def test_impliedvol():
    sigma = 0.2
    underlying_price = 100.
    strike = 90.
    maturity = 0.5

    option_price = blackscholes.formula(underlying_price, strike, maturity,
                                        sigma)
    iv = blackscholes.implied_vol(underlying_price, strike, maturity,
                                  option_price)

    assert abs(sigma - iv) < 1e-6


def test_impliedvol_exception():
    underlying_price = 100.
    strike = 90.
    maturity = 0.5

    option_price = 9.
    with raises(ValueError):
        blackscholes.implied_vol(underlying_price, strike, maturity,
                                 option_price)

    option_price = 101.
    with raises(ValueError):
        blackscholes.implied_vol(underlying_price, strike, maturity,
                                 option_price)


def test_vega():
    sigma = 0.2
    underlying_price = 100.
    strike = 90.
    maturity = 0.5

    vega_exact = blackscholes.vega(underlying_price, strike, maturity, sigma)
    vega_finite_diffs = 500.*(
        blackscholes.formula(underlying_price, strike, maturity, sigma + .001)
        - blackscholes.formula(underlying_price, strike, maturity, sigma - .001))

    assert abs(vega_exact - vega_finite_diffs) < 1e-3
