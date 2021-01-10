r"""
01. Automatic Differentiation
=============================
This tutorial focuses on one of the two main benefits of re-implementing
some of PyLops linear operators within the PyTorch framework, namely the
possibility to perform Automatic Differentiation (AD) on chains of operators
which can be:

- native PyTorch mathematical operations (e.g., :func:`torch.log`,
  :func:`torch.sin`, :func:`torch.tan`, :func:`torch.pow`, ...)
- neural network operators in :mod:`torch.nn`
- PyLops and/or PyLops-gpu linear operators

This opens up many opportunities, such as easily including linear regularization
terms to nonlinear cost functions or using linear preconditioners with nonlinear
modelling operators.

"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import gradcheck

import pylops_gpu
from pylops_gpu.utils.backend import device

dev = device()
plt.close('all')
np.random.seed(10)
torch.manual_seed(10)

###############################################################################
# In this example we consider a simple multidimensional functional:
#
# .. math::
#   \mathbf{y} = \mathbf{A} sin(\mathbf{x})
#
# and we use AD to compute the gradient with respect to the input vector
# evaluated at :math:`\mathbf{x}=\mathbf{x}_0` :
# :math:`\mathbf{g} = d\mathbf{y} / d\mathbf{x} |_{\mathbf{x}=\mathbf{x}_0}`.
#
# Let's start by defining the Jacobian:
#
#   .. math::
#        \textbf{J} = \begin{bmatrix}
#        dy_1 / dx_1 & ... & dy_1 / dx_M \\
#        ... & ... & ... \\
#        dy_N / dx_1 & ... & dy_N / dx_M
#        \end{bmatrix} = \begin{bmatrix}
#        a_{11} cos(x_1) & ... & a_{1M} cos(x_M) \\
#        ... & ... & ... \\
#        a_{N1} cos(x_1) & ... & a_{NM} cos(x_M)
#        \end{bmatrix} = \textbf{A} cos(\mathbf{x})
#
# Since both input and output are multidimensional,
# PyTorch ``backward`` actually computes the product between the transposed
# Jacobian and a vector :math:`\mathbf{v}`:
# :math:`\mathbf{g}=\mathbf{J^T} \mathbf{v}`.
#
# To validate the correctness of the AD result, we can in this simple case
# also compute the Jacobian analytically and apply it to the same vector
# :math:`\mathbf{v}` that we have provided to PyTorch ``backward``.

nx, ny = 10, 6
x0 = torch.arange(nx, dtype=torch.double, requires_grad=True)

# Forward
A = torch.normal(0., 1., (ny, nx), dtype=torch.double)
Aop = pylops_gpu.TorchOperator(pylops_gpu.MatrixMult(A))
y = Aop.apply(torch.sin(x0))

# AD
v = torch.ones(ny, dtype=torch.double)
y.backward(v, retain_graph=True)
adgrad = x0.grad

# Analytical
J = (A * torch.cos(x0))
anagrad = torch.matmul(J.T, v)

print('Input: ', x0)
print('AD gradient: ', adgrad)
print('Analytical gradient: ', anagrad)


###############################################################################
# Similarly we can use the :func:`torch.autograd.gradcheck` directly from
# PyTorch. Note that doubles must be used for this to succeed with very small
# `eps` and `atol`
input = (torch.arange(nx, dtype=torch.double, requires_grad=True),
         Aop.matvec, Aop.rmatvec, Aop.pylops, Aop.device)
test = gradcheck(Aop.Top, input, eps=1e-6, atol=1e-4)
print(test)


###############################################################################
# Note that while matrix-vector multiplication could have been performed using
# the native PyTorch operator :func:`torch.matmul`, in this case we have shown
# that we are also able to use a PyLops-gpu operator wrapped in
# :class:`pylops_gpu.TorchOperator`. As already mentioned, this gives us the
# ability to use much more complex linear operators provided by PyLops within
# a chain of mixed linear and nonlinear AD-enabled operators.
