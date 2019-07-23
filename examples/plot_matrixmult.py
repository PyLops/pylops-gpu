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

dev = device()
print('PyLops-gpu working on %s...' % dev)
plt.close('all')


###############################################################################
# Let's define the size ``N`` of thesquare matrix :math:`\mathbf{A}` and
# fill the matrix with random numbers

N = 20
A = torch.randn((N, N), dtype=torch.float32).to(dev)
Aop = pylops_gpu.MatrixMult(A, dtype='float64')

x = torch.ones(N, dtype=torch.float32).to(dev)

###############################################################################
# We can now apply the forward operator to create the data vector :math:`\mathbf{y}`
# and use ``/`` to solve the system by means of an explicit solver.
y = Aop*x
xest = cg(Aop, y, niter=N)[0]

###############################################################################
# Let's visually plot the system of equations we just solved.
gs = pltgs.GridSpec(1, 6)
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(gs[0, 0])
ax.imshow(y[:, np.newaxis], cmap='rainbow')
ax.set_title('y', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 1])
ax.text(0.35, 0.5, '=', horizontalalignment='center',
        verticalalignment='center', size=40, fontweight='bold')
ax.axis('off')
ax = plt.subplot(gs[0, 2:5])
ax.imshow(Aop.A, cmap='rainbow')
ax.set_title('A', size=20, fontweight='bold')
ax.set_xticks(np.arange(N-1)+0.5)
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 5])
ax.imshow(x[:, np.newaxis], cmap='rainbow')
ax.set_title('x', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

gs = pltgs.GridSpec(1, 6)
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(gs[0, 0])
ax.imshow(x[:, np.newaxis], cmap='rainbow')
ax.set_title('xest', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 1])
ax.text(0.35, 0.5, '=', horizontalalignment='center',
        verticalalignment='center', size=40, fontweight='bold')
ax.axis('off')
ax = plt.subplot(gs[0, 2:5])
ax.imshow(Aop.inv(), cmap='rainbow')
ax.set_title(r'A$^{-1}$', size=20, fontweight='bold')
ax.set_xticks(np.arange(N-1)+0.5)
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax = plt.subplot(gs[0, 5])
ax.imshow(y[:, np.newaxis], cmap='rainbow')
ax.set_title('y', size=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks(np.arange(N-1)+0.5)
ax.grid(linewidth=3, color='white')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
