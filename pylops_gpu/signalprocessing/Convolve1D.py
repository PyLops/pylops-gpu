import torch
import numpy as np

from torch.nn.functional import pad
from pylops_gpu import LinearOperator


class Convolve1D(LinearOperator):
    r"""1D convolution operator.

    Apply one-dimensional convolution with a compact filter to model (and data)
    along a specific direction of a multi-dimensional array depending on the
    choice of ``dir``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    h : :obj:`torch.Tensor`
        1d compact filter to be convolved to input signal
    offset : :obj:`int`
        Index of the center of the compact filter
    dims : :obj:`tuple`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which convolution is applied
    zero_edges : :obj:`bool`, optional
        Zero output at edges (`True`) or not (`False`)
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
    Refer to :class:`pylops.signalprocessing.Convolve1D` for implementation
    details.

    """
    def __init__(self, N, h, offset=0, dims=None, dir=0, zero_edges=False,
                 device='cpu', togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float32):
        self.nh = h.size()[0]
        self.h = h.reshape(1, 1, self.nh)
        self.offset = 2*(self.nh // 2 - int(offset))
        if self.offset != 0:
            self.h = pad(self.h, (self.offset if self.offset > 0 else 0,
                                  -self.offset if self.offset < 0 else 0),
                         mode='constant')
        self.padding = int(self.nh // 2 + np.abs(self.offset) // 2)
        self.dir = dir
        if dims is None:
            self.dims = (N, )
            self.reshape = False
        elif len(dims) == 1:
            self.dims = dims
            self.reshape = False
        else:
            if np.prod(dims) != N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = tuple(dims)
                self.otherdims = list(dims)
                self.otherdims.pop(self.dir)
                self.otherdims_prod = np.prod(self.dims) // self.dims[self.dir]
                self.dims_permute = list(self.dims)
                self.dims_permute[self.dir], self.dims_permute[-1] = \
                    self.dims_permute[-1], self.dims_permute[self.dir]
                self.dims_permute = tuple(self.dims_permute)
                self.permute = np.arange(0, len(self.dims))
                self.permute[self.dir], self.permute[-1] = \
                    self.permute[-1], self.permute[self.dir]
                self.permute = tuple(self.permute)
                self.reshape = True
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.zero_edges = zero_edges
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.dtype = dtype
        self.explicit = False
        self.Op = None

    def _matvec(self, x):
        if not self.reshape:
            x = x.reshape(1, 1, self.dims[0])
            y = torch.torch.conv_transpose1d(x, self.h, padding=self.padding)
            if self.zero_edges:
                y[..., :self.nh // 2] = 0
                y[..., -self.nh // 2 + 1:] = 0
        else:
            x1 = x.clone() # need to clone to avoid modifying x
            x1 = torch.reshape(x1, self.dims).permute(self.permute)
            y = torch.torch.conv_transpose1d(x1.reshape(self.otherdims_prod, 1,
                                                        self.dims[self.dir]),
                                             self.h, padding=self.padding)
            if self.zero_edges:
                y[..., :self.nh // 2] = 0
                y[..., -self.nh // 2 + 1:] = 0
            y = y.reshape(self.dims_permute).permute(self.permute)
        y = y.flatten()
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            x = x.reshape(1, 1, self.dims[0])
            if self.zero_edges:
                x[..., :self.nh // 2] = 0
                x[..., -self.nh // 2 + 1:] = 0
            y = torch.torch.conv1d(x, self.h, padding=self.padding)

        else:
            x1 = x.clone() # need to clone to avoid modifying x
            x1 = torch.reshape(x1, self.dims).permute(self.permute)
            if self.zero_edges:
                x1[..., :self.nh // 2] = 0
                x1[..., -self.nh // 2 + 1:] = 0
            y = torch.torch.conv1d(x1.reshape(self.otherdims_prod, 1,
                                              self.dims[self.dir]),
                                   self.h, padding=self.padding)
            y = y.reshape(self.dims_permute).permute(self.permute)
        y = y.flatten()
        return y
