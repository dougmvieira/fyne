import numpy as np
import pandas as pd
from fyne import blackscholes, heston


underlying_price = 100.
vol, kappa, theta, nu, rho = 0.0457, 5.07, 0.0457, 0.48, -0.767
strikes = pd.Index(np.linspace(80., 120., 40), name='Strike')
expiries = pd.Index([0.1, 0.3, 1.0], name='Expiry')

option_prices = heston.formula(underlying_price, strikes[:, None], expiries,
                               vol, kappa, theta, nu, rho)

implied_vols = pd.DataFrame(
    blackscholes.implied_vol(underlying_price, strikes[:, None], expiries,
                             option_prices),
    strikes, expiries)

implied_vols.plot().set_ylabel('Implied volatility')
