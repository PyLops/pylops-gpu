import torch

from pylops_gpu import Diagonal
from pylops_gpu.optimization.cg import cg


def NormalEquationsInversion(Op, Regs, data, Weight=None, dataregs=None,
                             epsI=0, epsRs=None, x0=None,
                             returninfo=False, **kwargs_cg):
    r"""Inversion of normal equations.

    Solve the regularized normal equations for a system of equations
    given the operator ``Op``, a data weighting operator ``Weight`` and
    a list of regularization terms ``Regs``

    Parameters
    ----------
    Op : :obj:`pylops_gpu.LinearOperator`
        Operator to invert
    Regs : :obj:`list`
        Regularization operators (``None`` to avoid adding regularization)
    data : :obj:`torch.Tensor`
        Data
    Weight : :obj:`pylops_gpu.LinearOperator`, optional
        Weight operator
    dataregs : :obj:`list`, optional
        Regularization data (must have the same number of elements
        as ``Regs``)
    epsI : :obj:`float`, optional
        Tikhonov damping
    epsRs : :obj:`list`, optional
         Regularization dampings (must have the same number of elements
         as ``Regs``)
    x0 : :obj:`torch.Tensor`, optional
        Initial guess
    returninfo : :obj:`bool`, optional
        Return info of CG solver
    **kwargs_cg
        Arbitrary keyword arguments for
        :py:func:`pylops_gpu.optimization.leastsquares.cg` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model.

    Notes
    -----
    Refer to :class:`pylops..optimization.leastsquares.NormalEquationsInversion`
    for implementation details.

    """
    dtype = data.dtype

    # store adjoint
    OpH = Op.H

    # create dataregs and epsRs if not provided
    if dataregs is None and Regs is not None:
        dataregs = [torch.zeros(Op.shape[1], dtype=dtype)] * len(Regs)

    if epsRs is None and Regs is not None:
        epsRs = [1] * len(Regs)

    # Normal equations
    if Weight is not None:
        y_normal = OpH * Weight * data
    else:
        y_normal = OpH * data
    if Weight is not None:
        Op_normal = OpH * Weight * Op
    else:
        Op_normal = OpH * Op

    # Add regularization terms
    if epsI > 0:
        Op_normal += epsI ** 2 * Diagonal(torch.ones(Op.shape[1]))

    if Regs is not None:
        for epsR, Reg, datareg in zip(epsRs, Regs, dataregs):
            RegH = Reg.H
            y_normal += epsR ** 2 * RegH * datareg
            Op_normal += epsR ** 2 * RegH * Reg

    # CG solver
    if x0 is not None:
        y_normal = y_normal - Op_normal * x0
    xinv, istop = cg(Op_normal, y_normal, **kwargs_cg)
    if x0 is not None:
        xinv = x0 + xinv

    if returninfo:
        return xinv, istop
    else:
        return xinv
