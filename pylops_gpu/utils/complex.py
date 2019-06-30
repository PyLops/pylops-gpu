import numpy as np
from pytorch_complex_tensor import ComplexTensor


def complextorch_fromnumpy(x):
    r"""Convert complex numpy array into torch ComplexTensor

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Numpy complex multi-dimensional array

    Returns
    -------
    xt : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor multi-dimensional array

    """
    xt = ComplexTensor(np.vstack((np.real(x), np.imag(x))))
    return xt


def complexnumpy_fromtorch(xt):
    r"""Convert complex numpy array into torch ComplexTensor

    Parameters
    ----------
    xt : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Numpy complex multi-dimensional array

    """
    x = xt.numpy()
    xrows = x.shape[0]
    x = x[:xrows//2] + 1j*x[xrows//2:]
    return x.squeeze()


def conj(x):
    r"""Apply complex conjugation to torch ComplexTensor

    Parameters
    ----------
    x : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Complex conjugated Torch ComplexTensor

    """
    xc = x.__graph_copy__(x.real, -x.imag)
    return xc