========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/fyne/badge/?style=flat
    :target: https://readthedocs.org/projects/fyne
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/dougmvieira/fyne.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/dougmvieira/fyne

.. |codecov| image:: https://codecov.io/github/dougmvieira/fyne/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/dougmvieira/fyne

.. |version| image:: https://img.shields.io/pypi/v/fyne.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/fyne

.. |commits-since| image:: https://img.shields.io/github/commits-since/dougmvieira/fyne/v0.4.0.svg
    :alt: Commits since latest release
    :target: https://github.com/dougmvieira/fyne/compare/v0.4.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/fyne.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/fyne

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/fyne.svg
    :alt: Supported versions
    :target: https://pypi.org/project/fyne

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/fyne.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/fyne


.. end-badges

The Python module for option pricing with affine models.

* Free software: MIT license

Installation
============

::

    pip install fyne

Documentation
=============


https://fyne.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
