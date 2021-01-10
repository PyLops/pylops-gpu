"""
Convolution
===========
This example shows how to use the :py:class:`pylops_gpu.signalprocessing.Convolve1D`
operator to perform convolution between two signals.

This example closely follow the equivalent

`PyLops example <https://pylops.readthedocs.io/en/latest/gallery/plot_convolve.html#sphx-glr-gallery-plot-convolve-py>`_.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pylops_gpu

from pylops.utils.wavelets import ricker
from pylops_gpu.utils.backend import device
from pylops_gpu.optimization.cg import cg

dev = device()
print('PyLops-gpu working on %s...' % dev)
plt.close('all')

###############################################################################
# We will start by creating a zero signal of length :math:`nt` and we will
# place a unitary spike at its center. We also create our filter to be
# applied by means of :py:class:`pylops_gpu.signalprocessing.Convolve1D`
# operator.
nt = 1001
dt = 0.004
t = np.arange(nt)*dt

x = torch.zeros(nt, dtype=torch.float32)
x[int(nt/2)] = 1

h, th, hcenter = ricker(t[:101], f0=30)
h = torch.from_numpy(h.astype(np.float32))
Cop = pylops_gpu.signalprocessing.Convolve1D(nt, h=h, offset=hcenter,
                                             dtype=torch.float32)
y = Cop*x

xinv = Cop / y

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x.cpu().numpy(), 'k', lw=2, label=r'$x$')
ax.plot(t, y.cpu().numpy(), 'r', lw=2, label=r'$y=Ax$')
ax.plot(t, xinv.cpu().numpy(), '--g', lw=2, label=r'$x_{ext}$')
ax.set_title('Convolve in 1st direction', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xlim(1.9, 2.1)

###############################################################################
# We show now that also a filter with mixed phase (i.e., not centered around zero)
# can be applied and inverted for using the :py:class:`pylops.signalprocessing.Convolve1D`
# operator.
Cop = pylops_gpu.signalprocessing.Convolve1D(nt, h=h, offset=hcenter - 3,
                                             dtype=torch.float32)
y = Cop * x
y1 = Cop.H * x
xinv = cg(Cop.H*Cop, Cop.H*y, niter=100)[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x.cpu().numpy(), 'k', lw=2, label=r'$x$')
ax.plot(t, y.cpu().numpy(), 'r', lw=2, label=r'$y=Ax$')
ax.plot(t, y1.cpu().numpy(), 'b', lw=2, label=r'$y=A^Hx$')
ax.plot(t, xinv.cpu().numpy(), '--g', lw=2, label=r'$x_{ext}$')
ax.set_title('Convolve in 1st direction', fontsize=14, fontweight='bold')
ax.set_xlim(1.9, 2.1)
ax.legend()