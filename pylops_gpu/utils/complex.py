import numpy as np
import torch
from pytorch_complex_tensor import ComplexTensor
from pytorch_complex_tensor.complex_scalar import ComplexScalar


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
    r"""Convert  torch ComplexTensor into complex numpy array

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


def complexscalar_fromtorchscalar(xt):
    r"""Convert torch ComplexScalar into complex number

    Parameters
    ----------
    xt : :obj:`pytorch_complex_scalar.ComplexScalar`
        Torch ComplexScalar

    Returns
    -------
    x : :obj:`complex`
        Complex scalar

    """
    x = xt.real.item() +1j*xt.imag.item()
    return x


def conj(x):
    r"""Apply complex conjugation to torch ComplexTensor

    Parameters
    ----------
    x : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor

    Returns
    -------
    x : :obj:`pytorch_complex_tensor.ComplexTensor`
        Complex conjugated Torch ComplexTensor

    """
    xc = x.__graph_copy__(x.real, -x.imag)
    return xc


def divide(x, y):
    r"""Element-wise division of  torch Tensor and torch ComplexTensor.

    Divide each element of ``x`` and ``y``, where one or both of them
    can contain complex numbers.

    Parameters
    ----------
    x : :obj:`pytorch_complex_tensor.ComplexTensor` or :obj:`torch.Tensor`
        Numerator
    y : :obj:`pytorch_complex_tensor.ComplexTensor`
        Denominator

    Returns
    -------
    div : :obj:`pytorch_complex_tensor.ComplexTensor`
        Complex conjugated Torch ComplexTensor

    """
    # convert to numpy
    if isinstance(x, ComplexTensor):
        xn = complexnumpy_fromtorch(x)
    elif isinstance(x, ComplexScalar):
        xn = complexscalar_fromtorchscalar(x)
    else:
        xn = x.cpu().numpy()
    if isinstance(y, ComplexTensor):
        yn = complexnumpy_fromtorch(y)
    elif isinstance(y, ComplexScalar):
        yn = complexscalar_fromtorchscalar(y)
    else:
        yn = y.cpu().numpy()
    # divide
    divn = xn / yn
    # convert back to torch
    if divn.size == 1:
        divn = divn.item()
    else:
        if np.iscomplexobj(divn):
            divn = complextorch_fromnumpy(divn)
        else:
            divn = torch.from_numpy(divn)
    return divn


def reshape(x, shape):
    r"""Reshape torch ComplexTensor

    Parameters
    ----------
    x : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor
    shape : :obj:`tuple`
        New shape

    Returns
    -------
    xreshaped : :obj:`pytorch_complex_tensor.ComplexTensor`
        Reshaped Torch ComplexTensor

    """
    xreshaped = x.reshape([2] + list(shape))
    xreshaped = ComplexTensor(np.vstack((xreshaped[0], xreshaped[1])))
    return xreshaped


def flatten(x):
    r"""Flatten torch ComplexTensor

    Parameters
    ----------
    x : :obj:`pytorch_complex_tensor.ComplexTensor`
        Torch ComplexTensor

    Returns
    -------
    xflattened : :obj:`pytorch_complex_tensor.ComplexTensor`
        Flattened Torch ComplexTensor

    """
    xflattened = ComplexTensor(np.vstack((x.real.view(-1),
                                          x.imag.view(-1))))
    return xflattened
