import torch
import numpy as np

from pylops_gpu.utils.complex import complextorch_fromnumpy


def dottest(Op, nr, nc, tol=1e-6, dtype=torch.float32,
            complexflag=0, device='cpu', raiseerror=True, verb=False):
    r"""Dot test.

    Generate random vectors :math:`\mathbf{u}` and :math:`\mathbf{v}`
    and perform dot-test to verify the validity of forward and adjoint operators.
    This test can help to detect errors in the operator implementation.

    Parameters
    ----------
    Op : :obj:`torch.Tensor`
        Linear operator to test.
    nr : :obj:`int`
        Number of rows of operator (i.e., elements in data)
    nc : :obj:`int`
        Number of columns of operator (i.e., elements in model)
    tol : :obj:`float`, optional
        Dottest tolerance
    dtype : :obj:`torch.dtype`, optional
        Type of elements in random vectors
    complexflag : :obj:`bool`, optional
        generate random vectors with real (0) or complex numbers
        (1: only model, 2: only data, 3:both)
    device : :obj:`str`, optional
        Device to be used
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity

    Raises
    ------
    ValueError
        If dot-test is not verified within chosen tolerance.

    Notes
    -----
    A dot-test is mathematical tool used in the development of numerical
    linear operators.

    More specifically, a correct implementation of forward and adjoint for
    a linear operator should verify the the following *equality*
    within a numerical tolerance:

    .. math::
        (\mathbf{Op}*\mathbf{u})^H*\mathbf{v} =
        \mathbf{u}^H*(\mathbf{Op}^H*\mathbf{v})

    """
    np_dtype = torch.ones(1, dtype=torch.float32).numpy().dtype
    if complexflag in (0, 2):
        u = torch.randn(nc, dtype=dtype)
    else:
        u = complextorch_fromnumpy(np.random.randn(nc).astype(np_dtype) +
                                   1j*np.random.randn(nc).astype(np_dtype))

    if complexflag in (0, 1):
        v = torch.randn(nr, dtype=dtype)
    else:
        v = complextorch_fromnumpy(np.random.randn(nr).astype(np_dtype) + \
                                   1j*np.random.randn(nr).astype(np_dtype))
    u, v = u.to(device), v.to(device)

    y = Op.matvec(u)   # Op * u
    x = Op.rmatvec(v)  # Op'* v

    if complexflag == 0:
        yy = torch.dot(y, v) # (Op  * u)' * v
        xx = torch.dot(u, x) # u' * (Op' * v)
    else:
        yy = np.vdot(y, v) # (Op  * u)' * v
        xx = np.vdot(u, x) # u' * (Op' * v)

    if complexflag == 0:
        if torch.abs((yy-xx)/((yy+xx+1e-15)/2)) < tol:
            if verb: print('Dot test passed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                                 % (yy, xx))
            if verb: print('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return False
    else:
        checkreal = np.abs((np.real(yy) - np.real(xx)) /
                           ((np.real(yy) + np.real(xx)+1e-15) / 2)) < tol
        checkimag = np.abs((np.real(yy) - np.real(xx)) /
                           ((np.real(yy) + np.real(xx)+1e-15) / 2)) < tol

        if checkreal and checkimag:
            if verb: print('Dot test passed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^H(Opu)=%f - u^H(Op^Hv)=%f'
                                 % (yy, xx))
            if verb: print('Dot test failed, v^H(Opu)=%f - u^H(Op^Hv)=%f'
                           % (yy, xx))
            return False
