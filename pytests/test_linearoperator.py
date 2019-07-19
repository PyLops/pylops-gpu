import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops_gpu.utils.backend import device
from pylops_gpu.utils.torch2numpy import *
from pylops_gpu import Diagonal, MatrixMult

par1 = {'ny': 11, 'nx': 11,
        'dtype': torch.float32}  # square
par2 = {'ny': 21, 'nx': 11,
        'dtype': torch.float32}  # overdetermined

dev = device()
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize("par", [(par1)])
def test_LinearOperator_fromnumpy(par):
    """Pass numpy to linear operator in forward and adjoint mode and check that
    matvec and rmatvec converts it to numpy
    """
    d = torch.arange(0, par['ny'], dtype=par['dtype']).to(dev)
    Dop = Diagonal(d)

    x = np.ones(par['ny'], dtype=numpytype_from_torchtype(par['dtype']))
    y = Dop.matvec(x)
    xadj = Dop.rmatvec(y)
    assert isinstance(y, np.ndarray)
    assert isinstance(xadj, np.ndarray)

    y = Dop * x
    xadj = Dop.H * x
    assert isinstance(y, np.ndarray)
    assert isinstance(xadj, np.ndarray)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_LinearOperator_adjoint(par):
    """Adjoint operator
    """
    A = np.random.randn(par['ny'], par['nx']).astype(np.float32)
    Aop = MatrixMult(torch.from_numpy(A))

    x = torch.ones(par['ny'], dtype=par['dtype']).to(dev)
    y = torch.from_numpy(A).t().matmul(x)
    y1 = Aop.rmatvec(x)
    y2 = Aop.H * x
    assert_array_equal(y.cpu().numpy(), y1.cpu().numpy())
    assert_array_equal(y.cpu().numpy(), y2.cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_LinearOperator_sum(par):
    """Sum of operators
    """
    d = torch.arange(0, par['ny'], dtype=par['dtype']).to(dev)
    d1 = torch.arange(10, par['ny'] + 10, dtype=par['dtype']).to(dev)
    Dop = Diagonal(d)
    D1op = Diagonal(d1)

    x = torch.ones(par['ny'], dtype=par['dtype']).to(dev)
    y = (Dop + D1op) * x
    y1 = d * x + d1 * x
    assert_array_equal(y.cpu().numpy(), y1.cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_LinearOperator_prod(par):
    """Product of operators
    """
    d = torch.arange(0, par['ny'], dtype=par['dtype']).to(dev)
    d1 = torch.arange(10, par['ny'] + 10, dtype=par['dtype']).to(dev)
    Dop = Diagonal(d)
    D1op = Diagonal(d1)
    Dprodop = D1op * Dop

    x = torch.ones(par['ny'], dtype=par['dtype']).to(dev)
    y = d * d1 * x
    y1 = D1op.matvec(Dop.matvec(x))
    y2 = Dprodop * x
    assert_array_equal(y.cpu().numpy(), y1.cpu().numpy())
    assert_array_equal(y.cpu().numpy(), y2.cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_LinearOperator_power(par):
    """Power operators - (Op ** 2) * x == Op * Op * x
    """
    d = torch.arange(0, par['ny'], dtype=par['dtype']).to(dev)
    Dop = Diagonal(d)

    x = torch.ones(par['ny'], dtype=par['dtype']).to(dev)
    y = Dop * (Dop * x)
    y1 = (Dop ** 2) * x
    assert_array_equal(y.cpu().numpy(), y1.cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_LinearOperator_div(par):
    """Division / to solve
    """
    d = torch.arange(1, par['ny'] + 1, dtype=par['dtype']).to(dev)
    Dop = Diagonal(d)

    x = torch.ones(par['ny'], dtype=par['dtype']).to(dev)
    y = Dop * x
    xinv = Dop / y
    assert_array_almost_equal(x.cpu().numpy(), xinv.cpu().numpy(),
                              decimal=3)
    