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
    FirstDerivative

Signal processing
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu.signalprocessing

.. autosummary::
   :toctree: generated/

    Convolve1D


Solvers
-------

Least-squares
~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu.optimization

.. autosummary::
   :toctree: generated/

    leastsquares.cg

