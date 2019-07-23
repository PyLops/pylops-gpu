import pytest
import torch
import numpy as np

from numpy.testing import assert_array_almost_equal

from pylops_gpu.utils import dottest
from pylops_gpu import MatrixMult, VStack
from pylops_gpu.optimization.cg import cg

par1 = {'ny': 101, 'nx': 101, 'imag': 0} # square real
par2 = {'ny': 301, 'nx': 101, 'imag': 0} # overdetermined real


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_VStack(par):
    """Dot-test and inversion for VStack operator
    """
    np.random.seed(10)
    G1 = torch.from_numpy(np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32'))
    G2 = torch.from_numpy(np.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32'))
    x = torch.ones(par['nx'], dtype=torch.float32) + \
        par['imag']*torch.ones(par['nx'], dtype=torch.float32)

    Vop = VStack([MatrixMult(G1, dtype=torch.float32),
                  MatrixMult(G2, dtype=torch.float32)],
                 dtype=torch.float32)
    assert dottest(Vop, 2*par['ny'], par['nx'],
                   complexflag=0 if par['imag'] == 0 else 3)

    xcg = cg(Vop.H * Vop, Vop.H * (Vop * x), niter=300)[0]
    assert_array_almost_equal(x.numpy(), xcg.numpy(), decimal=4)
