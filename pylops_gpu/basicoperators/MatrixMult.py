import torch
import numpy as np

from pytorch_complex_tensor import ComplexTensor
from pylops_gpu.LinearOperator import LinearOperator
from pylops_gpu.utils.complex import conj, reshape, flatten
from pylops_gpu.utils.torch2numpy import numpytype_from_torchtype


class MatrixMult(LinearOperator):
    r"""Matrix multiplication.

    Simple wrapper to :py:func:`torch.matmul` for
    an input matrix :math:`\mathbf{A}`.

    Parameters
    ----------
    A : :obj:`torch.Tensor` or :obj:`pytorch_complex_tensor.ComplexTensor` or :obj:`numpy.ndarray`
        Matrix.
    dims : :obj:`tuple`, optional
        Number of samples for each other dimension of model
        (model/data will be reshaped and ``A`` applied multiple times
        to each column of the model/data).
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
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.MatrixMult` for
    implementation details.

    """
    def __init__(self, A, dims=None, device='cpu',
                 togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float32):
        if not isinstance(A, (torch.Tensor, ComplexTensor)):
            self.complex = True if np.iscomplexobj(A) else False
            self.A = \
                torch.from_numpy(A.astype(numpytype_from_torchtype(dtype))).to(device)
        else:
            self.complex = True if isinstance(A, ComplexTensor) else False
            self.A = A
        if dims is None:
            self.reshape = False
            self.shape = A.shape
        else:
            if isinstance(dims, int):
                dims = (dims, )
            self.reshape = True
            self.dims = np.array(dims, dtype=np.int)
            self.shape = (A.shape[0]*np.prod(self.dims),
                          A.shape[1]*np.prod(self.dims))
            self.newshape = \
                (tuple(np.insert([np.prod(self.dims)], 0, self.A.shape[1])),
                 tuple(np.insert([np.prod(self.dims)], 0, self.A.shape[0])))

        self.complex = True if isinstance(A, ComplexTensor) else False
        if self.complex:
            self.Ac = conj(A).t()
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.dtype = dtype
        self.explicit = True
        self.Op = None

    def _matvec(self, x):
        if self.reshape:
            x = reshape(x, self.newshape[0]) if self.complex else \
                torch.reshape(x, self.newshape[0])
        else:
            if self.complex:
                x = x.t()
        if self.complex:
            y = self.A.mm(x)
            if not self.reshape:
                y = y.t()
        else:
            y = self.A.matmul(x)
        if self.reshape:
            y = flatten(y) if self.complex else y.view(-1)
        return y

    def _rmatvec(self, x):
        if self.reshape:
            x = reshape(x, self.newshape[1]) if self.complex else \
                torch.reshape(x, self.newshape[1])
        else:
            if self.complex:
                x = x.t()
        if self.complex:
            y = self.Ac.mm(x)
            if not self.reshape:
                y = y.t()
        else:
            y = self.A.t().matmul(x)
        if self.reshape:
            y = flatten(y) if self.complex else y.view(-1)
        return y

    def inv(self):
        r"""Return the inverse of :math:`\mathbf{A}`.

        Returns
        ----------
        Ainv : :obj:`torch.Tensor`
            Inverse matrix.

        """
        Ainv = torch.inverse(self.A)
        return Ainv


def aslinearoperator(A, device='cpu'):
    """Return A as a LinearOperator.

    ``A`` may be already a :class:`pylops_gpu.LinearOperator` or a
    :obj:`torch.Tensor`.

    """
    if isinstance(A, LinearOperator):
        return A
    else:
        return MatrixMult(A, device=device)
