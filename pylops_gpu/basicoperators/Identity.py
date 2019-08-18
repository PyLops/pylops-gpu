import torch
import numpy as np

from pytorch_complex_tensor import ComplexTensor
from pylops_gpu import LinearOperator
from pylops_gpu.utils.torch2numpy import numpytype_from_torchtype
from pylops_gpu.utils.complex import complextorch_fromnumpy

_complextypes = (torch.complex32, torch.complex64, torch.complex128)


class Identity(LinearOperator):
    r"""Identity operator.

    Simply move model to data in forward model and viceversa in adjoint mode if
    :math:`M = N`. If :math:`M > N` removes last :math:`M - N` elements from
    model in forward and pads with :math:`0` in adjoint. If :math:`N > M`
    removes last :math:`N - M` elements from data in adjoint and pads with
    :math:`0` in forward.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in data (and model, if ``M`` is not provided).
    M : :obj:`int`, optional
        Number of samples in model.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).
    complex : :obj:`bool`, optional
        Input model and data are complex arrays
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    dtype : :obj:`torch.dtype`, optional
        Type of elements in input array (if ``complex=True``, provide the
        type of the real component of the array)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Identity` for implementation
    details.

    """
    def __init__(self, N, M=None, inplace=True, complex=False, device='cpu',
                 togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float32):
        M = N if M is None else M
        self.inplace = inplace
        self.shape = (N, M)
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.dtype = dtype
        self.npdtype = numpytype_from_torchtype(self.dtype)
        self.complex = complex
        self.explicit = False
        self.Op = None

    def _matvec(self, x):
        if not self.inplace:
            if self.complex:
                x = x.__graph_copy__(x.real, x.imag)
            else:
                x = x.clone()
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            if self.complex:
                y = x[:, :self.shape[0]]
            else:
                y = x[:self.shape[0]]
        else:
            if self.complex:
                y = complextorch_fromnumpy(np.zeros(self.shape[0],
                                                    dtype=self.npdtype))
                y[:, :self.shape[1]] = x
            else:
                y = torch.zeros(self.shape[0], dtype=self.dtype)
                y[:self.shape[1]] = x
        return y

    def _rmatvec(self, x):
        if not self.inplace:
            if self.complex:
                x = x.__graph_copy__(x.real, x.imag)
            else:
                x = x.clone()
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            if self.complex:
                y = complextorch_fromnumpy(np.zeros(self.shape[1],
                                                    dtype=self.npdtype))
                y[:, :self.shape[0]] = x
            else:
                y = torch.zeros(self.shape[1], dtype=self.dtype)
                y[:self.shape[0]] = x
        else:
            if self.complex:
                y = x[:, :self.shape[1]]
            else:
                y = x[:self.shape[1]]
        return y
