import pytest

import numpy as np
import torch

from numpy.testing import assert_array_almost_equal
from pylops_gpu.utils import dottest
from pylops_gpu.utils.backend import device
from pylops_gpu.basicoperators import Diagonal
from pylops_gpu.optimization.leastsquares import cg


par1 = {'ny': 21, 'nx': 11, 'nt': 20,
        'dtype': torch.float32}  # real

dev = device()
np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_Diagonal_1dsignal(par):
    """Dot-test and inversion for Diagonal operator for 1d signal
    """
    for ddim in (par['nx'], par['nt']):
        d = (torch.arange(0, ddim, dtype=par['dtype']) + 1.).to(dev)

        Dop = Diagonal(d, dtype=par['dtype'])
        assert dottest(Dop, ddim, ddim)

        x = torch.ones(ddim, dtype=par['dtype']).to(dev)
        xcg = cg(Dop, Dop * x, niter=20)[0]
        assert_array_almost_equal(x.numpy(), xcg.numpy(), decimal=4)


@pytest.mark.parametrize("par", [(par1)])
def test_Diagonal_2dsignal(par):
    """Dot-test and inversion for Diagonal operator for 2d signal
    """
    for idim, ddim in enumerate((par['nx'], par['nt'])):
        d = (torch.arange(0, ddim, dtype=par['dtype']) + 1.).to(dev)

        Dop = Diagonal(d, dims=(par['nx'], par['nt']),
                       dir=idim, dtype=par['dtype'])
        assert dottest(Dop, par['nx']*par['nt'], par['nx']*par['nt'])

        x = torch.ones((par['nx'], par['nt']), dtype=par['dtype']).to(dev)
        xcg = cg(Dop, Dop * x.view(-1), niter=20)[0]

        assert_array_almost_equal(x.view(-1).numpy(), xcg.numpy(), decimal=4)
        