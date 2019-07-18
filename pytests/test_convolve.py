import pytest

import numpy as np
import torch

from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.signal import triang
from pylops.signalprocessing import Convolve1D

from pylops_gpu.utils.backend import device
from pylops_gpu.utils import dottest
from pylops_gpu.signalprocessing import Convolve1D  as gConvolve1D
from pylops_gpu.optimization.leastsquares import cg

# filters
nfilt = (5, 7)
h1 = torch.from_numpy(triang(nfilt[0], sym=True).astype(np.float32))
h2 = torch.from_numpy(np.outer(triang(nfilt[0], sym=True),
                               triang(nfilt[1], sym=True)).astype(np.float32))

par1_1d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': nfilt[0] // 2, 'dir': 0}  # zero phase, first direction
par2_1d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': 0, 'dir': 0}  # non-zero phase, first direction
par3_1d = {'nz': 21, 'ny': 51, 'nx': 51,
           'offset': nfilt[0] // 2, 'dir': 1}  # zero phase, second direction
par4_1d = {'nz': 21, 'ny': 61, 'nx': 51,
           'offset': nfilt[0] // 2 - 1,
           'dir': 1}  # non-zero phase, second direction

dev = device()


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d)])
def test_Convolve1D(par):
    """Dot-test, comparison with pylops and and inversion for Convolve1D
    operator
    """
    np.random.seed(10)
    #1D
    if par['dir'] == 0:
        Cop = gConvolve1D(par['nx'], h=h1, offset=par['offset'],
                         dtype=torch.float32)
        assert dottest(Cop, par['nx'], par['nx'])

        x = torch.zeros((par['nx']), dtype=torch.float32)
        x[par['nx']//2] = 1.

        Cop_ = Convolve1D(par['nx'], h=h1.cpu().numpy(), offset=par['offset'],
                          dtype='float32')
        assert_array_equal(Cop * x, Cop_ * x.cpu().numpy())

        if par['offset'] == nfilt[0]//2:
            # zero phase
            xcg = cg(Cop, Cop * x, niter=100)[0]
        else:
            # non-zero phase
            xcg = cg(Cop.H * Cop, Cop.H * (Cop * x), niter=100)[0]
        assert_array_almost_equal(x, xcg, decimal=1)

    # 1D on 2D
    Cop = gConvolve1D(par['ny'] * par['nx'], h=h1, offset=par['offset'],
                      dims=(par['ny'], par['nx']), dir=par['dir'],
                      dtype=torch.float32)
    assert dottest(Cop, par['ny'] * par['nx'], par['ny'] * par['nx'])

    x = torch.zeros((par['ny'], par['nx']), dtype=torch.float32)
    x[int(par['ny']/2-3):int(par['ny']/2+3),
      int(par['nx']/2-3):int(par['nx']/2+3)] = 1.
    x = x.flatten()

    Cop_ = Convolve1D(par['ny'] * par['nx'], h=h1.cpu().numpy(),
                      offset=par['offset'],
                      dims=(par['ny'], par['nx']), dir=par['dir'],
                      dtype='float32')
    assert_array_equal(Cop * x, Cop_ * x.cpu().numpy())

    if par['offset'] == nfilt[0] // 2:
        # zero phase
        xcg = cg(Cop, Cop * x, niter=100)[0]
    else:
        # non-zero phase
        xcg = cg(Cop.H * Cop, Cop.H * (Cop * x), niter=100)[0]
    assert_array_almost_equal(x, xcg, decimal=1)
