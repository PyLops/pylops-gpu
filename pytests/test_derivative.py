import pytest
import numpy as np
import torch

from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops.basicoperators import FirstDerivative, SecondDerivative, Laplacian

from pylops_gpu.utils.backend import device
from pylops_gpu.utils import dottest
from pylops_gpu import FirstDerivative as gFirstDerivative
from pylops_gpu import SecondDerivative as gSecondDerivative
from pylops_gpu import Laplacian as gLaplacian


par1 = {'nz': 10, 'ny': 30, 'nx': 40,
        'dz': 1., 'dy': 1., 'dx': 1.} # even with unitary sampling
par2 = {'nz': 10, 'ny': 30, 'nx': 40,
        'dz': 0.4, 'dy': 2., 'dx': 0.5} # even with non-unitary sampling
par3 = {'nz': 11, "ny": 51, 'nx': 61,
        'dz': 1., 'dy': 1., 'dx': 1.} # odd with unitary sampling
par4 = {'nz': 11, "ny": 51, 'nx': 61,
        'dz': 0.4, 'dy': 2., 'dx': 0.5} # odd with non-unitary sampling
par1e = {'nz': 10, 'ny': 30, 'nx': 40,
         'dz': 1., 'dy': 1., 'dx': 1.}  # even with unitary sampling
par2e = {'nz': 10, 'ny': 30, 'nx': 40,
         'dz': 0.4, 'dy': 2., 'dx': 0.5}  # even with non-unitary sampling
par3e = {'nz': 11, "ny": 51, 'nx': 61,
         'dz': 1., 'dy': 1., 'dx': 1.}  # odd with unitary sampling
par4e = {'nz': 11, "ny": 51, 'nx': 61,
         'dz': 0.4, 'dy': 2., 'dx': 0.5}  # odd with non-unitary sampling

dev = device()
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1e), (par2e), (par3e), (par4e)])
def test_FirstDerivative(par):
    """Dot-test and forward for FirstDerivative operator
    """
    # 1d
    gD1op = gFirstDerivative(par['nx'], sampling=par['dx'],
                             dtype=torch.float32)
    assert dottest(gD1op, par['nx'], par['nx'], tol=1e-3)

    x = torch.from_numpy((par['dx']*np.arange(par['nx'],
                                              dtype='float32')) ** 2)
    D1op = FirstDerivative(par['nx'], sampling=par['dx'],
                           dtype='float32')
    assert_array_equal((gD1op * x)[1:-1], (D1op * x.cpu().numpy())[1:-1])

    # 2d - derivative on 1st direction
    gD1op = gFirstDerivative(par['ny']*par['nx'], dims=(par['ny'], par['nx']),
                             dir=0, sampling=par['dy'], dtype=torch.float32)
    assert dottest(gD1op, par['ny']*par['nx'], par['ny']*par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dy']*np.arange(par['ny']))**2,
                                  np.ones(par['nx']))).astype(dtype='float32'))
    D1op = FirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                           dir=0, sampling=par['dy'], dtype='float32')
    gy = (gD1op * x.view(-1)).reshape(par['ny'], par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['ny'], par['nx'])
    assert_array_equal(gy[1:-1], y[1:-1])

    # 2d - derivative on 2nd direction
    gD1op = gFirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                             dir=1, sampling=par['dy'], dtype=torch.float32)
    assert dottest(gD1op, par['ny'] * par['nx'],
                   par['ny'] * par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dy'] * np.arange(par['ny'])) ** 2,
                                   np.ones(par['nx']))).astype(dtype='float32'))
    D1op = FirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                           dir = 1, sampling = par['dy'], dtype = 'float32')
    gy = (gD1op * x.view(-1)).reshape(par['ny'], par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['ny'], par['nx'])
    assert_array_equal(gy[:, 1:-1], y[:, 1:-1])

    # 3d - derivative on 1st direction
    gD1op = gFirstDerivative(par['nz'] * par['ny'] * par['nx'],
                             dims=(par['nz'], par['ny'], par['nx']),
                             dir=0, sampling=par['dz'], dtype=torch.float32)
    assert dottest(gD1op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dz']*np.arange(par['nz']))**2,
                                   np.ones((par['ny'],
                                            par['nx']))).astype(dtype='float32')))
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=0, sampling=par['dz'], dtype='float32')

    gy = (gD1op * x.view(-1)).reshape(par['nz'], par['ny'], par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['nz'], par['ny'],
                                                  par['nx'])
    assert_array_almost_equal(gy[1:-1], y[1:-1], decimal=5)

    # 3d - derivative on 2nd direction
    gD1op = gFirstDerivative(par['nz'] * par['ny'] * par['nx'],
                             dims=(par['nz'], par['ny'], par['nx']),
                             dir=1, sampling=par['dy'], dtype=torch.float32)
    assert dottest(gD1op, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dz'] * np.arange(par['nz'])) ** 2,
                                   np.ones((par['ny'],
                                            par['nx']))).reshape(par['nz'],
                                                                 par['ny'],
                                                                 par['nx'])).astype(dtype='float32'))
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=1, sampling=par['dy'], dtype='float32')

    gy = (gD1op * x.view(-1)).reshape(par['nz'], par['ny'],
                                      par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['nz'], par['ny'],
                                                  par['nx'])
    assert_array_almost_equal(gy[1:-1], y[1:-1],
                              decimal=5)

    # 3d - derivative on 3rd direction
    gD1op = gFirstDerivative(par['nz']*par['ny']*par['nx'],
                             dims=(par['nz'], par['ny'], par['nx']),
                             dir=2, sampling=par['dx'], dtype=torch.float32)
    assert dottest(gD1op, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dz'] * np.arange(par['nz'])) ** 2,
                                   np.ones((par['ny'],
                                            par['nx']))).reshape(par['nz'],
                                                                 par['ny'],
                                                                 par['nx'])).astype(dtype='float32'))

    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=2, sampling=par['dx'], dtype='float32')

    gy = (gD1op * x.view(-1)).reshape(par['nz'], par['ny'],
                                      par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['nz'], par['ny'],
                                                  par['nx'])
    assert_array_almost_equal(gy[1:-1], y[1:-1], decimal=5)



@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1e), (par2e), (par3e), (par4e)])
def test_SecondDerivative(par):
    """Dot-test and forward for SecondDerivative operator
    """
    # 1d
    gD1op = gSecondDerivative(par['nx'], sampling=par['dx'],
                              dtype=torch.float32)
    assert dottest(gD1op, par['nx'], par['nx'], tol=1e-3)

    x = torch.from_numpy((par['dx']*np.arange(par['nx'],
                                              dtype='float32')) ** 2)
    D1op = SecondDerivative(par['nx'], sampling=par['dx'],
                            dtype='float32')
    assert_array_equal((gD1op * x)[1:-1], (D1op * x.cpu().numpy())[1:-1])

    # 2d - derivative on 1st direction
    gD1op = gSecondDerivative(par['ny']*par['nx'], dims=(par['ny'], par['nx']),
                              dir=0, sampling=par['dy'], dtype=torch.float32)
    assert dottest(gD1op, par['ny']*par['nx'], par['ny']*par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dy']*np.arange(par['ny']))**2,
                                  np.ones(par['nx']))).astype(dtype='float32'))
    D1op = SecondDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                            dir=0, sampling=par['dy'], dtype='float32')
    gy = (gD1op * x.view(-1)).reshape(par['ny'], par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['ny'], par['nx'])
    assert_array_equal(gy[1:-1], y[1:-1])

    # 2d - derivative on 2nd direction
    gD1op = gSecondDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                              dir=1, sampling=par['dy'], dtype=torch.float32)
    assert dottest(gD1op, par['ny'] * par['nx'],
                   par['ny'] * par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dy'] * np.arange(par['ny'])) ** 2,
                                   np.ones(par['nx']))).astype(dtype='float32'))
    D1op = SecondDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                            dir = 1, sampling = par['dy'], dtype = 'float32')
    gy = (gD1op * x.view(-1)).reshape(par['ny'], par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['ny'], par['nx'])
    assert_array_equal(gy[:, 1:-1], y[:, 1:-1])

    # 3d - derivative on 1st direction
    gD1op = gSecondDerivative(par['nz'] * par['ny'] * par['nx'],
                              dims=(par['nz'], par['ny'], par['nx']),
                              dir=0, sampling=par['dz'], dtype=torch.float32)
    assert dottest(gD1op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dz']*np.arange(par['nz']))**2,
                                   np.ones((par['ny'],
                                            par['nx']))).astype(dtype='float32')))
    D1op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['nz'], par['ny'], par['nx']),
                            dir=0, sampling=par['dz'], dtype='float32')

    gy = (gD1op * x.view(-1)).reshape(par['nz'], par['ny'], par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['nz'], par['ny'],
                                                  par['nx'])
    assert_array_almost_equal(gy[1:-1], y[1:-1], decimal=5)

    # 3d - derivative on 2nd direction
    gD1op = gSecondDerivative(par['nz'] * par['ny'] * par['nx'],
                              dims=(par['nz'], par['ny'], par['nx']),
                              dir=1, sampling=par['dy'], dtype=torch.float32)
    assert dottest(gD1op, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dz'] * np.arange(par['nz'])) ** 2,
                                   np.ones((par['ny'],
                                            par['nx']))).reshape(par['nz'],
                                                                 par['ny'],
                                                                 par['nx'])).astype(dtype='float32'))
    D1op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['nz'], par['ny'], par['nx']),
                            dir=1, sampling=par['dy'], dtype='float32')

    gy = (gD1op * x.view(-1)).reshape(par['nz'], par['ny'],
                                      par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['nz'], par['ny'],
                                                  par['nx'])
    assert_array_almost_equal(gy[1:-1], y[1:-1],
                              decimal=5)

    # 3d - derivative on 3rd direction
    gD1op = gSecondDerivative(par['nz']*par['ny']*par['nx'],
                              dims=(par['nz'], par['ny'], par['nx']),
                              dir=2, sampling=par['dx'], dtype=torch.float32)
    assert dottest(gD1op, par['nz']*par['ny']*par['nx'],
                   par['nz']*par['ny']*par['nx'], tol=1e-3)

    x = torch.from_numpy((np.outer((par['dz'] * np.arange(par['nz'])) ** 2,
                                   np.ones((par['ny'],
                                            par['nx']))).reshape(par['nz'],
                                                                 par['ny'],
                                                                 par['nx'])).astype(dtype='float32'))

    D1op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['nz'], par['ny'], par['nx']),
                            dir=2, sampling=par['dx'], dtype='float32')

    gy = (gD1op * x.view(-1)).reshape(par['nz'], par['ny'],
                                      par['nx']).cpu().numpy()
    y = (D1op * x.view(-1).cpu().numpy()).reshape(par['nz'], par['ny'],
                                                  par['nx'])
    assert_array_almost_equal(gy[1:-1], y[1:-1], decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1e), (par2e), (par3e), (par4e)])
def test_Laplacian(par):
    """Dot-test for Laplacian operator
    """
    # 2d - symmetrical
    Dlapop = gLaplacian((par['ny'], par['nx']), dirs=(0, 1), weights=(1, 1),
                       sampling=(par['dy'], par['dx']), dtype=torch.float32)
    assert dottest(Dlapop, par['ny'] * par['nx'], par['ny'] * par['nx'],
                   tol=1e-3)

    # 2d - asymmetrical
    Dlapop = gLaplacian((par['ny'], par['nx']), dirs=(0, 1), weights=(1, 2),
                       sampling=(par['dy'], par['dx']), dtype=torch.float32)
    assert dottest(Dlapop, par['ny'] * par['nx'], par['ny'] * par['nx'],
                   tol=1e-3)

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = gLaplacian((par['nz'], par['ny'], par['nx']), dirs=(0, 1),
                        weights=(1, 1), sampling=(par['dy'], par['dx']),
                        dtype=torch.float32)
    assert dottest(Dlapop, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)

    # 3d - symmetrical on 1st and 2nd direction
    Dlapop = gLaplacian((par['nz'], par['ny'], par['nx']), dirs=(0, 1),
                        weights=(1, 1), sampling=(par['dy'], par['dx']),
                        dtype=torch.float32)
    assert dottest(Dlapop, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'], tol=1e-3)
