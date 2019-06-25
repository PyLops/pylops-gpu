import pytest

import numpy as np
import torch
from pytorch_complex_tensor import ComplexTensor

from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr

from pylops_gpu.utils import dottest
from pylops_gpu.utils.backend import device
from pylops_gpu.basicoperators import Diagonal


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
        dottest(Dop, ddim, ddim, verb=True)

        x = torch.ones(ddim, dtype=par['dtype']).to(dev)

        xlsqr = lsqr(Dop, Dop * x, damp=1e-20, iter_lim=300, show=0)[0]
        assert_array_almost_equal(x, xlsqr, decimal=4)
