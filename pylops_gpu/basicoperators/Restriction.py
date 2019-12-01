import torch
import numpy as np

from pylops_gpu.LinearOperator import LinearOperator


class Restriction(LinearOperator):
    r"""Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava``
    in forward mode and place those values at locations ``iava``
    in an otherwise zero vector in adjoint mode.

    Parameters
    ----------
    M : :obj:`int`
        Number of samples in model.
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Integer indices of available samples for data selection.
    dims : :obj:`list`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which restriction is applied.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    dtype : :obj:`torch.dtype`, optional
        Type of elements in input array.


    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Restriction` for
    implementation details.

    """
    def __init__(self, M, iava, dims=None, dir=0, inplace=True,
                 device='cpu', togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float32):
        self.M = M
        self.dir = dir
        self.iava = iava
        if dims is None:
            self.N = len(iava)
            self.dims = (self.M, )
            self.reshape = False
        else:
            if np.prod(dims) != self.M:
                raise ValueError('product of dims must equal M!')
            else:
                self.dims = dims # model dimensions
                self.dimsd = list(dims) # data dimensions
                self.dimsd[self.dir] = len(iava)
                self.iavareshape = [1] * self.dir + [len(self.iava)] + \
                                   [1] * (len(self.dims) - self.dir - 1)
                self.N = np.prod(self.dimsd)
                self.reshape = True
        self.inplace = inplace
        self.shape = (self.N, self.M)
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.dtype = dtype
        self.explicit = True
        self.Op = None

    def _matvec(self, x):
        if not self.inplace:
            x = x.copy()
        if not self.reshape:
            y = x[self.iava]
        else:
            raise NotImplementedError('Restriction currently works only on '
                                      '1d arrays')
        #    x = torch.reshape(x, self.dims)
        #    y = torch.take(x, self.iava, axis=self.dir)
        #    y = y.view(-1)
        return y

    def _rmatvec(self, x):
        if not self.inplace:
            x = x.copy()
        if not self.reshape:
            y = torch.zeros(self.dims, dtype=self.dtype).to(self.device)
            y[self.iava] = x
        else:
            raise NotImplementedError('Restriction currently works only on '
                                      '1d arrays')
        #    x = torch.reshape(x, self.dimsd)
        #    y = torch.zeros(self.dims, dtype=self.dtype)
        #    torch.put_along_axis(y, torch.reshape(self.iava, self.iavareshape),
        #                         x, axis=self.dir)
        #    y = y.view(-1)
        return y
