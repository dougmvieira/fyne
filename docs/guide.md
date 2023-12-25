# User guide

## Models

The models that are currently implemented in `fyne` are the Black-Scholes,
Heston and Wishart models. In order to make notation clear, especially with the
naming of the parameters, we state below the discounted underlying price
dynamics of the models under the risk-neutral measure.

-   Black-Scholes

\\[
dS_t = \sigma S_t dW_t
\\]

-   Heston

\begin{aligned}
dS_t      & = \sqrt{V_t} S_t dW_t \\
dV_t      & = \kappa(\theta - V_t)dt + \nu \sqrt{V_t}dZ_t \\
d[W, Z]_t & = \rho dt
\end{aligned}

-   Wishart

\begin{aligned}
dS_t      & = S_t \mathrm{Tr} \left( \sqrt{V_t} \left(
                  dW_t R + dZ_t \sqrt{I - RR^T} \right) \right) \\
dV_t      & = \left(\beta QQ^T + M V_t + V_t M^T \right) dt
            + \sqrt{V_t} dW_t Q + Q^T dW_t^T \sqrt{V_t} \\
d[W, Z]_t & = 0
\end{aligned}


## Pricing

Each model has its own pricing formula. The available pricing functions
are:

- `fyne.blackscholes.formula`
- `fyne.heston.formula`

These functions return the price of the option in monetary units. If
implied volatility is needed, it can be evaluated with
`fyne.blackscholes.implied_vol`{.interpreted-text role="func"}.

### Example

In this example, we compute the implied volatility smile according to
the Heston model.

```python
--8<-- "docs/heston_smile.py"
```
--8<-- "docs/heston_smile.html"

## Greeks

Greeks are usually associated to the derivatives of the Black-Scholes
formula. However, Greeks can be computed according to other models as
well. The following are the available Greeks in `fyne`:

- `fyne.blackscholes.delta`
- `fyne.blackscholes.vega`
- `fyne.heston.delta`
- `fyne.heston.vega`

## Calibration

In `fyne` we distinguish two types of calibration:

- Cross-sectional
  - Calibration from options prices at a single point in time
- Panel
  - Calibration from options prices at a multiple points in time

Besides, calibration can recover the full parameter set and unobservable
state variables or just the unobservable state variables.

The available calibration functions are the following:

- `fyne.heston.calibration_crosssectional`
- `fyne.heston.calibration_panel`
- `fyne.heston.calibration_vol`
