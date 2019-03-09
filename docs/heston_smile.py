import numpy as np
import pandas as pd
from fyne import blackscholes, heston


underlying_price = 100.
vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
strikes = pd.Index(np.linspace(80., 120., 40), name='Strike')
expiries = pd.Index([0.1, 0.3, 1.0], name='Expiry')

option_prices = pd.DataFrame([[heston.formula(underlying_price, strike, expiry,
                                              vol, kappa, theta, nu, rho)
                               for expiry in expiries] for strike in strikes],
                             strikes, expiries)
implied_vols = option_prices.apply(
    lambda smile: [blackscholes.implied_vol(underlying_price, strike,
                                            smile.name, price)
                   for strike, price in zip(smile.index, smile)])

implied_vols.plot().set_ylabel('Implied volatility')
