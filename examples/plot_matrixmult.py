r"""
Matrix Multiplication
=====================

This example shows how to use the :py:class:`pylops_gpu.MatrixMult` operator
to perform *Matrix inversion* of the following linear system.

.. math::
        \mathbf{y}=  \mathbf{A} \mathbf{x}

For square :math:`\mathbf{A}`, we will use the
:py:func:`pylops_gpu.optimization.leastsquares.cg` solver.

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs
import pylops_gpu

from pylops_gpu.utils.backend import device
from pylops_gpu.optimization.cg import cg

torch.manual_seed(0)
dev = device()
print('PyLops-gpu working on %s...' % dev)
plt.close('all')


###############################################################################
# Let's define the size ``N`` of thesquare matrix :math:`\mathbf{A}` and
# fill the matrix with random numbers
N = 20
A = torch.randn((N, N), dtype=torch.float32).to(dev)
A = torch.matmul(A.t(), A) # need semi-definite positive matrix for cg
Aop = pylops_gpu.MatrixMult(A, dtype=torch.float32)

x = torch.ones(N, dtype=torch.float32).to(dev)

###############################################################################
# We can now apply the forward operator to create the data vector :math:`\mathbf{y}`
# and use ``/`` to solve the system by means of an explicit solver.
# If you prefer to customize the solver (e.g., choosing the number of
# iterations) use the method ``div`` instead.
y = Aop * x
xest = Aop / y
xest = Aop.div(y, niter=2*N)

print('x', x)
print('xest', xest)