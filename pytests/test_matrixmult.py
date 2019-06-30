import pytest
import torch

from numpy.testing import assert_array_almost_equal
from pylops_gpu.utils import dottest
from pylops_gpu.utils.backend import device
from pylops_gpu.utils.complex import *
from pylops_gpu import MatrixMult
from pylops_gpu.optimization.leastsquares import cg


par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'dtype': 'float32'}  # square real
#par2 = {'ny': 21, 'nx': 11, 'imag': 0,
#        'dtype': 'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11, 'imag': 1j,
         'dtype': 'float32'}  # square complex
#par2j = {'ny': 21, 'nx': 11, 'imag': 1j,
#         'dtype': 'float32'}  # overdetermined complex

dev = device()
np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_MatrixMult(par):
    """Dot-test and inversion for MatrixMult operator
    """
    np.random.seed(10)
    G = np.random.normal(0, 10, (par['ny'],
                                 par['nx'])).astype(par['dtype']) + \
        par['imag']*np.random.normal(0, 10, (par['ny'],
                                             par['nx'])).astype(par['dtype'])
    if par['imag'] == 0:
        G = np.dot(G.T, G)
        G = torch.from_numpy(G).to(dev)
    else:
        G = np.dot(G.conj().T, G)
        G = complextorch_fromnumpy(G).to(dev)
    Gop = MatrixMult(G, dtype=G.dtype)
    assert dottest(Gop, par['ny'], par['nx'], tol=1e-4,
                   complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx'], dtype='float32') + \
        par['imag']*np.ones(par['nx'], dtype='float32')
    if par['imag'] == 0:
        x = torch.from_numpy(x).to(dev)
    else:
        x = complextorch_fromnumpy(x).to(dev)
    y = Gop * x
    xcg = cg(Gop, y, niter=2*par['nx'])[0]
    assert_array_almost_equal(x.numpy(), xcg.numpy(), decimal=3)
