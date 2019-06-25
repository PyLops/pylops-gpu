import numpy as np
from pytorch_complex_tensor import ComplexTensor


def complextorch_fromnumpy(x):
    r"""Convert complex numpy array into torch ComplexTensor

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Numpy complex array

    Returns
    -------
    xt : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor array

    """
    xt = ComplexTensor([np.real(x), np.imag(x)])
    return xt


def complexnumpy_fromtorch(xt):
    r"""Convert complex numpy array into torch ComplexTensor

    Parameters
    ----------
    xt : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor array

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Numpy complex array

    """
    x = xt.numpy()
    x = x[0] + 1j*x[1]
    return x