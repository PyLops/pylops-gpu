r"""
FISTA
=====

This example shows how to use the
:py:class:`pylops_gpu.optimization.sparsity.FISTA` solver.

This solver can be used when the model to retrieve is supposed to have
a sparse representation in a certain domain. FISTA solves an
uncostrained problem with a L1 regularization term:

.. math::
    J = ||\mathbf{d} - \mathbf{Op} \mathbf{x}||_2 + \epsilon ||\mathbf{x}||_1

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pylops
import pylops_gpu

from pylops_gpu.utils.backend import device

dev = device()
print('PyLops-gpu working on %s...' % dev)
plt.close('all')

torch.manual_seed(0)
np.random.seed(1)
dtype = torch.float32

###############################################################################
# Let's start with a simple example, where we create a dense mixing matrix
# and a sparse signal and we use OMP and ISTA to recover such a signal.
# Note that the mixing matrix leads to an underdetermined system of equations
# (:math:`N < M`) so being able to add some extra prior information regarding
# the sparsity of our desired model is essential to be able to invert
# such a system.

N, M = 15, 20
A = np.random.randn(N, M).astype(np.float32)
Aop = pylops_gpu.MatrixMult(torch.from_numpy(A), device=dev)

x = torch.from_numpy(np.random.rand(M).astype(np.float32))
x[x < 0.9] = 0
y = Aop * x

# FISTA
eps = 0.5
maxit = 1000
x_fista = pylops_gpu.optimization.sparsity.FISTA(Aop, y, maxit, eps=eps,
                                                 tol=1e-10)[0]

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.stem(x, linefmt='k', basefmt='k',
        markerfmt='ko', label='True')
ax.stem(x_fista, linefmt='--r',
        markerfmt='ro', label='FISTA')
ax.set_title('Model', size=15, fontweight='bold')
ax.legend()
plt.tight_layout()


###############################################################################
# We now consider a more interesting problem problem, *wavelet deconvolution*
# from a signal that we assume being composed by a train of spikes convolved
# with a certain wavelet. We will see how solving such a problem with a
# least-squares solver such as
# :py:class:`pylops_gpu.optimization.leastsquares.RegularizedInversion` does
# not produce the expected results (especially in the presence of noisy data),
# conversely using the :py:class:`pylops_gpu.optimization.sparsity.FISTA`
# solver allows us to succesfully retrieve the input signal even
# in the presence of noise.

nt = 61
dt = 0.004
t = np.arange(nt)*dt
x = np.zeros(nt, dtype=np.float32)
x[10] = -.4
x[int(nt/2)] = 1
x[nt-20] = 0.5
x = torch.from_numpy(x)

h, th, hcenter = pylops.utils.wavelets.ricker(t[:101], f0=20)
h = torch.from_numpy(h.astype(np.float32))
Cop = pylops_gpu.signalprocessing.Convolve1D(nt, h=h, offset=int(hcenter),
                                             dtype=dtype)
y = Cop * x

xls = pylops_gpu.optimization.cg.cg(Cop, y, niter=10,  tol=1e-10)[0]

xfista = \
    pylops_gpu.optimization.sparsity.FISTA(Cop, y, niter=400, eps=5e-1,
                                           tol=1e-8)[0]

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(t, x, 'k', lw=8, label=r'$x$')
ax.plot(t, y, 'r', lw=4, label=r'$y=Ax$')
ax.plot(t, xls, '--g', lw=4, label=r'$x_{LS}$')
ax.plot(t, xfista, '--m', lw=4, label=r'$x_{FISTA}$')
ax.set_title('Deconvolution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
