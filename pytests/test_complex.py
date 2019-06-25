import numpy as np

from numpy.testing import assert_array_equal
from pylops_gpu.utils.complex import *


def test_complex_numpy_torch_numpy():
    """Switch between numpy and torch ComplexTensor
    """
    x = np.ones(5) + 3j * np.ones(5)
    xt = complextorch_fromnumpy(x)
    xback = complexnumpy_fromtorch(xt)
    assert_array_equal(x, xback)


def test_complex_attrs():
    """Compare attributes of numpy complex and torch ComplexTensor
    """
    x = np.ones(5, dtype=np.float32) + 3j * np.ones(5, dtype=np.float32)
    y = 2*np.ones(5, dtype=np.float32) - 1j * np.ones(5, dtype=np.float32)
    sum = x + y
    sub = x - y
    mul = x * y

    xt = complextorch_fromnumpy(x)
    yt = complextorch_fromnumpy(y)
    sumt = xt + yt
    subt = xt - yt
    mult = xt * yt

    assert_array_equal(np.abs(x), xt.abs().numpy().squeeze()) # abs
    assert_array_equal(sum, complexnumpy_fromtorch(sumt)) # sum
    assert_array_equal(sub, complexnumpy_fromtorch(subt)) # sub
    assert_array_equal(mul, complexnumpy_fromtorch(mult)) # mul

