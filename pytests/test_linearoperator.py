import pytest

import numpy as np
import torch

from pylops_gpu.utils.backend import device
from pylops_gpu.utils.torch2numpy import *
from pylops_gpu.basicoperators import Diagonal

par1 = {'ny': 11, 'nx': 11,
        'dtype': torch.float32}  # square
par2 = {'ny': 21, 'nx': 11,
        'dtype': torch.float32}  # overdetermined

dev = device()
np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_LinearOperator_fromnumpy(par):
    """Pass numpy to linear operator in forward and adjoint mode and check that
    matve and rmatvec converts it to numpy
    """
    d = torch.arange(0, par['ny'], dtype=par['dtype']).to(dev)
    Dop = Diagonal(d)

    x = np.ones(par['ny'], dtype=numpytype_from_torchtype(par['dtype']))
    y = Dop.matvec(x)
    xadj = Dop.rmatvec(y)

    assert isinstance(y, np.ndarray)
    assert isinstance(xadj, np.ndarray)
