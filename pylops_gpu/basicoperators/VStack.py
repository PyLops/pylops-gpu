import torch
import numpy as np

from scipy.sparse.linalg.interface import _get_dtype
from pylops_gpu.LinearOperator import LinearOperator


class VStack(LinearOperator):
    r"""Vertical stacking.

    Stack a set of N linear operators vertically.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    dtype : :obj:`str`, optional
        Type of elements in input array

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.VStack` for
    implementation details.

    """
    def __init__(self, ops, device='cpu', togpu=(False, False),
                 tocpu=(False, False), dtype=torch.float32):
        self.ops = ops
        nops = np.zeros(len(ops), dtype=np.int)
        for iop, oper in enumerate(ops):
            nops[iop] = oper.shape[0]
        self.nops = nops.sum()
        self.mops = ops[0].shape[1]
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.shape = (self.nops, self.mops)
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.dtype = dtype
        self.explicit = False
        self.Op = None

    def _matvec(self, x):
        y = torch.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.nnops[iop]:self.nnops[iop + 1]] = oper.matvec(x).squeeze()
        return y

    def _rmatvec(self, x):
        y = torch.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y += oper.rmatvec(x[self.nnops[iop]:self.nnops[iop + 1]]).squeeze()
        return y
