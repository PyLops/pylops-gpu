import torch

from pytorch_complex_tensor import ComplexTensor
from pylops_gpu.utils.complex import divide
#from pylops_gpu import LinearOperator, aslinearoperator


def cg(A, y, x=None, niter=10, tol=1e-10):
    r"""Conjugate gradient

    Solve a system of equations given the square operator ``A`` and data ``y``
    using conjugate gradient iterations.

    Parameters
    ----------
    A : :obj:`pylops_gpu.LinearOperator`
        Operator to invert of size :math:`[N \times N]`
    y : :obj:`torch.Tensor`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`torch.Tensor`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    tol : :obj:`int`, optional
        Residual norm tolerance

    Returns
    -------
    x : :obj:`torch.Tensor`
        Estimated model
    iiter : :obj:`torch.Tensor`
        Max number of iterations model

    """
    complex_problem = True if isinstance(y, ComplexTensor) else False
    #if not isinstance(A, LinearOperator):
    #    A = aslinearoperator(A)
    if x is None:
        if complex_problem:
            x = ComplexTensor(torch.zeros((2 * y.shape[-1], 1),
                                          dtype=y.dtype)).t()
        else:
            x = torch.zeros_like(y)
    r = y - A.matvec(x)
    c = r.clone()
    if complex_problem:
        c = ComplexTensor(c)
    kold = torch.sum(r * r)

    iiter = 0
    while iiter < niter and torch.abs(kold) > tol:
        Ac = A.matvec(c)
        cAc = (c * Ac).sum() if complex_problem else torch.sum(c * Ac)
        a = divide(kold, cAc) if complex_problem else kold / cAc
        x += a * c
        r -= a * Ac
        k = torch.sum(r * r)
        b = k / kold
        c = r + b * c
        kold = k
        iiter += 1
    return x, iiter


def cgls(A, y, x=None, niter=10, damp=0., tol=1e-10):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given an operator ``A`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    A : :obj:`pylops_gpu.LinearOperator`
        Operator to invert of size :math:`[N \times M]`
    y : :obj:`torch.Tensor`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`torch.Tensor`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    damp : :obj:`float`, optional
        Damping coefficient
    tol : :obj:`int`, optional
        Residual norm tolerance

    Returns
    -------
    x : :obj:`torch.Tensor`
        Estimated model
    iiter : :obj:`torch.Tensor`
        Max number of iterations model

    Notes
    -----
    Minimize the following functional using conjugate gradient
    iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Ax} ||^2 + \epsilon || \mathbf{x} ||^2

    where :math:`\epsilon` is the damping coefficient.
    """
    # naive approach ##
    # Op = A.H * A
    # y = A.H * y
    # return cg(Op, y, x=x, niter=niter, tol=tol)

    complex_problem = True if isinstance(y, ComplexTensor) else False
    # if not isinstance(A, LinearOperator):
    #    A = aslinearoperator(A)
    if x is None:
        if complex_problem:
            x = ComplexTensor(torch.zeros((2 * A.shape[1], 1),
                                          dtype=y.dtype)).t()
        else:
            x = torch.zeros(A.shape[1], dtype=y.dtype)
    s = y - A.matvec(x)
    r = A.rmatvec(s) - damp * x
    c = r.clone()
    if complex_problem:
        c = ComplexTensor(c)
    kold = torch.sum(r * r)
    q = A.matvec(c)
    iiter = 0
    while iiter < niter and torch.abs(kold) > tol:
        qq = (q * q).sum()
        a = divide(kold, qq) if complex_problem else kold / qq
        x += a * c
        s -= a * q
        r = A.rmatvec(s) - damp * x
        k = torch.sum(r * r) if complex_problem else torch.sum(r * r)
        b = k / kold
        c = r + b * c
        q = A.matvec(c)
        kold = k
        iiter += 1
    return x, iiter
