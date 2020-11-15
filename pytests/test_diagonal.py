import pytest
import torch

from numpy.testing import assert_array_almost_equal
from pylops_gpu.utils import dottest
from pylops_gpu.utils.backend import device
from pylops_gpu.utils.complex import *
from pylops_gpu.basicoperators import Diagonal
from pylops_gpu.optimization.cg import cg


par1 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float32'}  # real
par2 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'float32'}  # complex

dev = device()
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize("par", [(par1)])#, (par2)])
def test_Diagonal_1dsignal(par):
    """Dot-test and inversion for Diagonal operator for 1d signal
    """
    for ddim in (par['nx'], par['nt']):
        d = (np.arange(0, ddim, dtype=par['dtype']) + 1.) + \
            par['imag']*(np.arange(0, ddim, dtype=par['dtype']) + 1.)
        if par['imag'] == 0:
            d = torch.from_numpy(d).to(dev)
        else:
            d = complextorch_fromnumpy(d).to(dev)

        Dop = Diagonal(d, dtype=d.dtype)
        assert dottest(Dop, ddim, ddim, tol=1e-4,
                       complexflag=0 if par['imag'] == 0 else 3)

        x = np.ones(ddim, dtype=par['dtype']) + \
            par['imag'] * np.ones(ddim, dtype=par['dtype'])
        if par['imag'] == 0:
            x = torch.from_numpy(x).to(dev)
        else:
            x = complextorch_fromnumpy(x).to(dev)
        xcg = cg(Dop, Dop * x, niter=ddim)[0]
        assert_array_almost_equal(x.numpy(), xcg.cpu().numpy(), decimal=4)


@pytest.mark.parametrize("par", [(par1)])#, (par2)])
def test_Diagonal_2dsignal(par):
    """Dot-test and inversion for Diagonal operator for 2d signal
    """
    for idim, ddim in enumerate((par['nx'], par['nt'])):
        d = (np.arange(0, ddim, dtype=par['dtype']) + 1.) + \
            par['imag'] * (np.arange(0, ddim, dtype=par['dtype']) + 1.)
        if par['imag'] == 0:
            d = torch.from_numpy(d).to(dev)
        else:
            d = complextorch_fromnumpy(d).to(dev)

        Dop = Diagonal(d, dims=(par['nx'], par['nt']),
                       dir=idim, dtype=par['dtype'])
        assert dottest(Dop, par['nx']*par['nt'], par['nx']*par['nt'], tol=1e-4,
                       complexflag=0 if par['imag'] == 0 else 3)

        x = np.ones((par['nx'], par['nt']), dtype=par['dtype']) + \
            par['imag'] * np.ones((par['nx'], par['nt']), dtype=par['dtype'])
        if par['imag'] == 0:
            x = torch.from_numpy(x).to(dev)
        else:
            x = complextorch_fromnumpy(x).to(dev)
        xcg = cg(Dop, Dop * x.flatten(), niter=Dop.shape[0])[0]
        assert_array_almost_equal(x.flatten().numpy(),
                                  xcg.flatten().cpu().numpy(), decimal=4)
