PyLops-GPU
==========

.. note:: This library is under early development.

   Expect things to constantly change until version v1.0.0.

This library is an extension of `PyLops <https://pylops.readthedocs.io/en/latest/>`_
to run operators on GPUs.

As much as `numpy <http://www.numpy.org>`_ and
`scipy <http://www.scipy.org/scipylib/index.html>`_ lie at the core of the parent project
PyLops, PyLops-GPU heavily builds on top of `PyTorch <http://pytorch.org>`_
and takes advantage of the same optimized tensor computations used in PyTorch
for deep learning using GPUs and CPUs. Doing so, linear operators can be computed on GPUs.

Here is a simple example showing how a diagonal operator can be created,
applied and inverted using PyLops:

.. code-block:: python

   import numpy as np
   from pylops import Diagonal

   n = int(1e6)
   x = np.ones(n)
   d = np.arange(n) + 1.

   Dop = Diagonal(d)

   # y = Dx
   y = Dop*x

and similarly using PyLops-GPU:

.. code-block:: python

   import numpy as np
   import torch
   from pylops_gpu.utils.backend import device
   from pylops_gpu import Diagonal

   dev = device() # will return 'gpu' if GPU is available

   n = int(1e6)
   x = torch.ones(n, dtype=torch.float64).to(dev)
   d = (torch.arange(0, n, dtype=torch.float64) + 1.).to(dev)

   Dop = Diagonal(d, device=dev)

   # y = Dx
   y = Dop*x

Running these two snippets of code in Google Colab with GPU enabled gives a 50+
speed up for the forward pass.

As a by-product of implementing PyLops linear operators in PyTorch, we can easily
chain our operators with any nonlinear mathematical operation (e.g., log, sin, tan, pow, ...)
as well as with operators from the ``torch.nn`` submodule and obtain *Automatic
Differentiation* (AD) for the entire chain. Since the gradient of a linear
operator is simply its *adjoint*, we have implemented a single class,
:py:class:`pylops_gpu.TorchOperator`, which can wrap any linear operator
from PyLops and PyLops-gpu libraries and return a :py:class:`torch.autograd.Function` object.


History
-------
PyLops-GPU was initially written and it is currently maintained by `Equinor <https://www.equinor.com>`_
It is an extension of `PyLops <https://pylops.readthedocs.io/en/latest/>`_ for large-scale optimization with
*GPU*-powered linear operators that can be tailored to our needs, and as contribution to the free software community.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started:

   installation.rst
   tutorials/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation:

   api/index.rst
   api/others.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved:

   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Roadmap <roadmap.rst>
   Credits <credits.rst>

