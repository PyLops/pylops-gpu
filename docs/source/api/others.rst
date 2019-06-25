.. _others:


PyLops-GPU Utilities
====================
Alongside with its *Linear Operators* and *Solvers*, PyLops contains also a number of auxiliary routines
performing universal tasks that are used by several operators or simply within one or more :ref:`tutorials` for
the preparation of input data and subsequent visualization of results.

Shared
------

Backends
~~~~~~~~

.. currentmodule:: pylops_gpu.utils

.. autosummary::
   :toctree: generated/

    backend.device
    complex.complextorch_fromnumpy
    complex.complexnumpy_fromtorch
