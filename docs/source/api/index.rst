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
    TorchOperator


Basic operators
~~~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu

.. autosummary::
   :toctree: generated/

    MatrixMult
    Identity
    Diagonal
    VStack


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

.. currentmodule:: pylops_gpu.optimization.cg

.. autosummary::
   :toctree: generated/

    cg
    cgls

Least-squares
~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu.optimization

.. autosummary::
   :toctree: generated/

    leastsquares.NormalEquationsInversion

Sparsity
~~~~~~~~

.. autosummary::
   :toctree: generated/

    sparsity.FISTA
    sparsity.SplitBregman


Applications
------------

Geophysical subsurface characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    poststack.PoststackInversion