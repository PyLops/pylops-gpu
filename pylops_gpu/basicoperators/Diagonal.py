import torch
import numpy as np
from pytorch_complex_tensor import ComplexTensor

from pylops_gpu import LinearOperator


class Diagonal(LinearOperator):
    r"""Diagonal operator.

    Applies element-wise multiplication of the input vector with the vector
    ``diag`` in forward and with its complex conjugate in adjoint mode.

    This operator can also broadcast; in this case the input vector is
    reshaped into its dimensions ``dims`` and the element-wise multiplication
    with ``diag`` is perfomed on the direction ``dir``. Note that the
    vector ``diag`` will need to have size equal to ``dims[dir]``.

    Parameters
    ----------
    diag : :obj:`numpy.ndarray` or :obj:`torch.Tensor`
        Vector to be used for element-wise multiplication.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which multiplication is applied.
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Diagonal` for implementation
    details.

    """
    def __init__(self, diag, dims=None, dir=0, device='cpu',
                 togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float64):
        if not isinstance(diag, (torch.Tensor, ComplexTensor)):
            self.complex = True if np.iscomplexobj(self.diag) else False
            self.diag = torch.from_numpy(diag.flatten())
        else:
            self.complex = True if isinstance(diag, ComplexTensor) else False
            self.diag = diag.flatten()
        if dims is None:
            self.shape = (len(self.diag), len(self.diag))
            self.dims = None
            self.reshape = False
        else:
            diagdims = [1] * len(dims)
            diagdims[dir] = dims[dir]
            self.diag = self.diag.reshape(diagdims)
            self.shape = (np.prod(dims), np.prod(dims))
            self.dims = dims
            self.reshape = True
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.dtype = dtype
        self.explicit = False
        self.Op = None

    def _matvec(self, x):
        if not self.reshape:
            y = self.diag*x
        else:
            x = x.reshape(self.dims)
            y = (self.diag*x).flatten()
        return y

    def _rmatvec(self, x):
        if self.complex:
            diagadj = self.diag.conj()
        else:
            diagadj = self.diag
        if not self.reshape:
            y = diagadj * x
        else:
            x = x.reshape(self.dims)
            y = diagadj * x
        return y
