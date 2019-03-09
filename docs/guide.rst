User guide
==========

.. contents:: Contents
   :local:


Models
------

The models that are currently implemented in :mod:`fyne` are the Black-Scholes
and Heston models. In order to make notation clear, especially with the naming
of the parameters, we state below the discounted underlying price dynamics of
the models under the risk-neutral measure.

* Black-Scholes

.. math::

   dS_t = \sigma S_t dW_t

* Heston

.. math::

   dS_t      & = \sqrt{V_t} S_t dW_t \\
   dV_t      & = \kappa(\theta - V_t)dt + \nu \sqrt{V_t}dZ_t \\
   d[W, Z]_t & = \rho dt


Pricing
-------

Each model has its own pricing formula. The available pricing functions are:

* :func:`fyne.blackscholes.formula`
* :func:`fyne.heston.formula`

These functions return the price of the option in monetary units. If implied
volatility is needed, it can be evaluated with
:func:`fyne.blackscholes.implied_vol`.


Example
^^^^^^^

In this example, we compute the implied volatility smile according to the Heston
model.

.. plot:: heston_smile.py
   :include-source:


Greeks
------

Greeks are usually associated to the derivatives of the Black-Scholes formula.
However, Greeks can be computed according to other models as well. The following
are the available Greeks in :mod:`fyne`:

* :func:`fyne.blackscholes.delta`
* :func:`fyne.blackscholes.vega`
* :func:`fyne.heston.delta`
* :func:`fyne.heston.vega`


Calibration
-----------

In :mod:`fyne` we distinguish two types of calibration:

* Cross-sectional
  * Calibration from options prices at a single point in time
* Panel
  * Calibration from options prices at a multiple points in time

Besides, calibration can recover the full parameter set and unobservable state
variables or just the unobservable state variables.

The available calibration functions are the following:

* :func:`fyne.heston.calibration_crosssectional`
* :func:`fyne.heston.calibration_panel`
* :func:`fyne.heston.calibration_vol`
