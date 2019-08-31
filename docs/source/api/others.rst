.. _others:


PyLops-GPU Utilities
====================
Alongside with its *Linear Operators* and *Solvers*, PyLops-GPU contains
also a number of auxiliary routines.


Shared
------

Backends
~~~~~~~~

.. currentmodule:: pylops_gpu.utils

.. autosummary::
   :toctree: generated/

    backend.device


Dot-test
~~~~~~~~

.. currentmodule:: pylops_gpu.utils

.. autosummary::
   :toctree: generated/

    dottest


Torch2Numpy
~~~~~~~~~~~

.. currentmodule:: pylops_gpu.utils

.. autosummary::
   :toctree: generated/

    torch2numpy.numpytype_from_torchtype
    torch2numpy.torchtype_from_numpytype


Complex Tensors
~~~~~~~~~~~~~~~

.. currentmodule:: pylops_gpu.utils

.. autosummary::
   :toctree: generated/

    complex.complextorch_fromnumpy
    complex.complexnumpy_fromtorch
    complex.conj
    complex.divide
    complex.reshape
    complex.flatten

