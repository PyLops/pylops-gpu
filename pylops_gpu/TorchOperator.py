import torch
import numpy as np

from pylops import LinearOperator
from pylops_gpu import LinearOperator as gLinearOperator


class _TorchOperator(torch.autograd.Function):
    """Wrapper class for PyLops operators into Torch functions

    The flag pylops is used to discriminate pylops and pylops-gpu operators;
    the former one requires the input to be converted into numpy.ndarray and
    the output to be converted back to torch.Tensor

    """
    @staticmethod
    def forward(ctx, x, forw, adj, pylops):
        ctx.forw = forw
        ctx.adj = adj
        ctx.pylops = pylops

        if ctx.pylops:
            x = x.detach().numpy()
        y = ctx.forw(x)
        if ctx.pylops:
            y = torch.from_numpy(y)
        return y

    @staticmethod
    def backward(ctx, y):
        if ctx.pylops:
            y = y.detach().numpy()
        x = ctx.adj(y)
        if ctx.pylops:
            x = torch.from_numpy(x)
        return  x, None, None, None


class TorchOperator():
    """Wrap pylops operator into Torch function

    This class can be used to wrap a pylops (or pylops-gpu) operator into a
    torch function. Doing so, users can mix native torch functions (e.g.
    basic linear algebra operations, neural networks, etc.) and pylops
    operators.

    Since all operators in PyLops are linear operators, a Torch function is
    simply implemented by using the forward operator for its forward pass
    and the adjont operator for its backward (gradient) pass.

    Parameters
    ----------
    Op : :obj:`pylops_gpu.LinearOperator` or :obj:`pylops.LinearOperator`
        PyLops operator
    pylops : :obj:`bool`, optional
        ``Op`` is a pylops operator (``True``) or a pylops-gpu
        operator (``False``)
    Returns
    -------
    y : :obj:`torch.Tensor`
        Output array resulting from the application of the operator to ``x`

    """
    def __init__(self, Op, pylops=False):
        self.pylops = True if isinstance(Op, LinearOperator) else False
        self.matvec = Op.matvec
        self.rmatvec = Op.rmatvec

    def apply(self, x):
        """Apply forward pass to input vector

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            Input array

        Returns
        -------
        y : :obj:`torch.Tensor`
            Output array resulting from the application of the operator to ``x`

        """
        y = _TorchOperator.apply(x, self.matvec, self.rmatvec, self.pylops)
        return y
