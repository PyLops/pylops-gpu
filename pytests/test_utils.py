import pytest
import torch
import numpy as np

from numpy.testing import assert_array_equal
from pylops_gpu.utils.torch2numpy import torchtype_from_numpytype, \
    numpytype_from_torchtype
from pylops_gpu.utils.complex import *

par1 = {'dims': (5,)}  # 1d
par2 = {'dims': (5, 3)}  # 2d


def test_typeconversion():
    """Verify numpy to torch (and viceversa) type conversions
    """
    numpytypes = [np.float32, np.float64, np.int16, np.int32]
    torchtypes = [torch.float32, torch.float64, torch.int16, torch.int32]
    for numpytype, torchtype in zip(numpytypes, torchtypes):
        torchtype_check = torchtype_from_numpytype(numpytype)
        numpytype_check = numpytype_from_torchtype(torchtype)
        assert numpytype_check == numpytype
        assert torchtype_check == torchtype


"""
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_complex_attrs(par):
    #Compare attributes of numpy complex and torch ComplexTensor
    x = np.ones(par['dims'], dtype=np.float32) + \
        3j * np.ones(par['dims'], dtype=np.float32)
    y = 2*np.ones(par['dims'], dtype=np.float32) - \
        1j * np.ones(par['dims'], dtype=np.float32)
    sum = x + y
    sub = x - y
    mul = x * y
    xc = x.conjugate()

    xt = complextorch_fromnumpy(x)
    yt = complextorch_fromnumpy(y)
    sumt = xt + yt
    subt = xt - yt
    mult = xt * yt
    xct = conj(xt)
    xflattened = flatten(xt)


    assert_array_equal(np.abs(x), xt.abs().numpy().squeeze()) # abs
    assert_array_equal(sum, complexnumpy_fromtorch(sumt)) # sum
    assert_array_equal(sub, complexnumpy_fromtorch(subt)) # sub
    assert_array_equal(mul, complexnumpy_fromtorch(mult)) # mul
    assert_array_equal(xc, complexnumpy_fromtorch(xct)) # conj
    assert xflattened.shape[1] == np.prod(np.array(par['dims'])) # flatten
"""