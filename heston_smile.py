import numpy as np
import pandas as pd
import plotly.express as px

from fyne import blackscholes, heston

underlying_price = 100.0
vol = 0.0457
params = (5.07, 0.0457, 0.48, -0.767)
strikes, expiries = np.broadcast_arrays(
    np.linspace(80.0, 120.0, 40)[:, None],
    [0.1, 0.3, 1.0],
)

option_prices = heston.formula(
    underlying_price, strikes, expiries, vol, *params
)
implied_vols = blackscholes.implied_vol(
    underlying_price, strikes, expiries, option_prices
)

data = pd.DataFrame(
    {
        "expirie": expiries.ravel(),
        "strike": strikes.ravel(),
        "implied_vol": implied_vols.ravel(),
    }
)
fig = px.line(
    data,
    x="strike",
    y="implied_vol",
    color="expirie",
    title="Heston volatility smile",
    markers=True,
)

fig.write_html(
    "docs/heston_smile.html",
    full_html=False,
    include_plotlyjs="cdn",
)
