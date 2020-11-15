import pytest
import torch

from numpy.testing import assert_array_almost_equal
from pylops_gpu.utils import dottest
from pylops_gpu.utils.torch2numpy import torchtype_from_numpytype
from pylops_gpu.utils.backend import device
from pylops_gpu.utils.complex import *
from pylops_gpu.basicoperators import Identity
from pylops_gpu.optimization.cg import cg


par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'dtype': 'float32'}  # square real
par2 = {'ny': 21, 'nx': 11, 'imag': 0,
        'dtype': 'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11, 'imag': 1j,
         'dtype': 'float32'}  # square complex
par2j = {'ny': 21, 'nx': 11, 'imag': 1j,
         'dtype': 'float32'}  # overdetermined complex
par3 = {'ny': 11, 'nx': 21, 'imag': 0,
        'dtype': 'float32'}  # underdetermined real

dev = device()
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])#, (par1j), (par2j)])
def test_Identity_inplace(par):
    """Dot-test, forward and adjoint for Identity operator
    """
    Iop = Identity(par['ny'], par['nx'],
                   complex=True if par['imag'] == 1j else False,
                   dtype=torchtype_from_numpytype(par['dtype']),
                   inplace=True)
    assert dottest(Iop, par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx'], dtype='float32') + \
        par['imag'] * np.ones(par['nx'], dtype='float32')
    if par['imag'] == 0:
        x = torch.from_numpy(x).to(dev)
    else:
        x = complextorch_fromnumpy(x).to(dev)

    y = Iop*x
    x1 = Iop.H*y

    if par['imag'] == 0:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        x1 = x1.cpu().numpy()
    else:
        x = complexnumpy_fromtorch(x)
        y = complexnumpy_fromtorch(y)
        x1 = complexnumpy_fromtorch(x1)

    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              y[:min(par['ny'], par['nx'])],
                              decimal=4)
    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              x1[:min(par['ny'], par['nx'])],
                              decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)]) # (par1j), (par2j),
def test_Identity_noinplace(par):
    """Dot-test, forward and adjoint for Identity operator (not in place)
    """
    print('complex', True if par['imag'] == 1j else False)
    Iop = Identity(par['ny'], par['nx'],
                   complex=True if par['imag'] == 1j else False,
                   dtype=torchtype_from_numpytype(par['dtype']),
                   inplace=False)
    assert dottest(Iop, par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx'], dtype='float32') + \
        par['imag'] * np.ones(par['nx'], dtype='float32')
    if par['imag'] == 0:
        x = torch.from_numpy(x).to(dev)
    else:
        x = complextorch_fromnumpy(x).to(dev)
    y = Iop*x
    x1 = Iop.H*y

    if par['imag'] == 0:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        x1 = x1.cpu().numpy()
    else:
        x = complexnumpy_fromtorch(x)
        y = complexnumpy_fromtorch(y)
        x1 = complexnumpy_fromtorch(x1)

    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              y[:min(par['ny'], par['nx'])],
                              decimal=4)
    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              x1[:min(par['ny'], par['nx'])],
                              decimal=4)

    # change value in x and check it doesn't change in y
    x[0] = 10
    assert x[0] != y[0]
