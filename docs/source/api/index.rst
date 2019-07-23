.. _api:

PyLops-GPU API
==============


Linear operators
----------------

Templates
~~~~~~~~~
.. automodule:: pylops_gpu

.. currentmodule:: pylops_gpu

.. autosummary::
   :toctree: generated/

    LinearOperator

Basic operators
~~~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu

.. autosummary::
   :toctree: generated/

    MatrixMult
    Diagonal


Smoothing and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   FirstDerivative
   SecondDerivative
   Laplacian


Signal processing
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu.signalprocessing

.. autosummary::
   :toctree: generated/

    Convolve1D


Solvers
-------

Low-level solvers
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu.optimization

.. autosummary::
   :toctree: generated/

    cg.cg

Least-squares
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    leastsquares.NormalEquationsInversion

