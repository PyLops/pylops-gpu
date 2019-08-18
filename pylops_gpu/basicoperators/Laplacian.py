import torch
import numpy as np

from pylops_gpu.basicoperators import SecondDerivative


def Laplacian(dims, dirs=(0, 1), weights=(1, 1), sampling=(1, 1),
              device='cpu', togpu=(False, False), tocpu=(False, False),
              dtype=torch.float32):
    r"""Laplacian.

    Apply second-order centered laplacian operator to a multi-dimensional
    array (at least 2 dimensions are required)

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    dirs : :obj:`tuple`, optional
        Directions along which laplacian is applied.
    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if
        ``weights=[1,1]``)
    sampling : :obj:`tuple`, optional
        Sampling steps ``dx`` and ``dy`` for each direction
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``)
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    l2op : :obj:`pylops.LinearOperator`
        Laplacian linear operator

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Laplacian` for implementation
    details.

    Note that since the Torch implementation is based on a convolution
    with a compact filter :math:`[1., -2., 1.]`, edges are treated
    differently compared to the PyLops equivalent operator.

    """
    l2op = weights[0]*SecondDerivative(np.prod(dims), dims=dims, dir=dirs[0],
                                       sampling=sampling[0], device=device,
                                       togpu=togpu, tocpu=tocpu, dtype=dtype)
    l2op += weights[1]*SecondDerivative(np.prod(dims), dims=dims, dir=dirs[1],
                                        sampling=sampling[1], device=device,
                                        togpu=togpu, tocpu=tocpu, dtype=dtype)
    return l2op
