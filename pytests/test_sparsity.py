import pytest
import torch

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops_gpu.utils.backend import device
from pylops_gpu.basicoperators import MatrixMult, Identity, FirstDerivative
from pylops_gpu.optimization.sparsity import FISTA, SplitBregman

par1 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': False,
        'dtype': 'float64'}  # square real, zero initial guess
par2 = {'ny': 11, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # square real, non-zero initial guess
par3 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0':False,
        'dtype':'float64'} # overdetermined real, zero initial guess
par4 = {'ny': 31, 'nx': 11, 'imag': 0, 'x0': True,
        'dtype': 'float64'} # overdetermined real, non-zero initial guess
par5 = {'ny': 21, 'nx': 41, 'imag': 0, 'x0': True,
        'dtype': 'float64'}  # underdetermined real, non-zero initial guess
par1j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': False,
         'dtype': 'complex64'}  # square complex, zero initial guess
par2j = {'ny': 11, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # square complex, non-zero initial guess
par3j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0':False,
         'dtype':'complex64'} # overdetermined complex, zero initial guess
par4j = {'ny': 31, 'nx': 11, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'} # overdetermined complex, non-zero initial guess
par5j = {'ny': 21, 'nx': 41, 'imag': 1j, 'x0': True,
         'dtype': 'complex64'}  # underdetermined complex, non-zero initial guess

dev = device()


@pytest.mark.parametrize("par", [(par1), (par3), (par5),
                                 (par1j), (par3j), (par5j)])
def test_FISTA(par):
    """Invert problem with FISTA
    """
    np.random.seed(42)
    A = np.random.randn(par['ny'], par['nx']).astype(np.float32)
    Aop = MatrixMult(torch.from_numpy(A).to(dev), device=dev)

    x = torch.zeros(par['nx'])
    x[par['nx'] // 2] = 1
    x[3] = 1
    x[par['nx'] - 4] = -1
    y = Aop * x

    eps = 0.5
    maxit = 2000

    # FISTA
    xinv, _, _, _, _ = FISTA(Aop, y, maxit, eps=eps, eigsiter=100,
                             tol=0, returninfo=True)
    assert_array_almost_equal(x.cpu().numpy(), xinv.cpu().numpy(), decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_SplitBregman(par):
    """Invert denoise problem with SplitBregman
    """
    np.random.seed(42)
    nx = 3 * par['nx']  # need enough samples for TV regularization to be effective
    Iop = Identity(nx)
    Dop = FirstDerivative(nx)

    x = torch.zeros(nx)
    x[:nx // 2] = 10
    x[nx // 2:3 * nx // 4] = -5
    n = torch.from_numpy(np.random.normal(0, 1, nx).astype(np.float32))
    y = x + n

    mu = 0.01
    lamda = 0.2
    niter_end = 100
    niter_in = 3
    x0 = torch.ones(nx)
    xinv, _ = SplitBregman(Iop, [Dop], y, niter_end, niter_in,
                           mu=mu, epsRL1s=[lamda],
                           tol=1e-4, tau=1,
                           x0=x0 if par['x0'] else None,
                           restart=False, **dict(niter=5))
    assert (np.linalg.norm(x.cpu().numpy() - xinv.cpu().numpy()) /
            np.linalg.norm(x.cpu().numpy())) < 1e-1
