import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops_gpu.utils.backend import device
from pylops_gpu.utils.torch2numpy import *
from pylops_gpu import TorchOperator, MatrixMult

par1 = {'ny': 11, 'nx': 11,
        'dtype': torch.float32}  # square
par2 = {'ny': 21, 'nx': 11,
        'dtype': torch.float32}  # overdetermined

dev = device()
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator(par):
    """Apply forward and gradient. As for linear operators the gradient
    must equal the adjoint of operator applied to the same vector, the two
    results are also checked to be the same.
    """
    Dop = MatrixMult(torch.randn(par['ny'], par['nx']), device=dev)
    Top = TorchOperator(Dop, batch=False, pylops=False)

    x = torch.randn(par['nx']).to(dev)
    xt = x.view(-1)
    xt.requires_grad = True
    v = torch.randn(par['ny']).to(dev)

    # pylops-gpu operator
    y = Dop * x
    xadj = Dop.H * v

    # torch operator
    yt = Top.apply(xt)
    yt.backward(v, retain_graph=True)

    assert_array_equal(y.detach().cpu().numpy(), yt.detach().cpu().numpy())
    assert_array_equal(xadj.detach().cpu().numpy(), xt.grad.cpu().numpy())


@pytest.mark.parametrize("par", [(par1)])
def test_TorchOperator_batch(par):
    """Apply forward for input with multiple samples (= batch)
    """
    Dop = MatrixMult(torch.randn(par['ny'], par['nx']), device=dev)
    Top = TorchOperator(Dop, batch=True, pylops=False)

    x = torch.randn((4, par['nx'])).to(dev)
    x1 = x.T

    y = Dop.matmat(x, kfirst=True)
    y1 = Dop.matmat(x1, kfirst=False)
    yt = Top.apply(x)

    assert_array_equal(y.cpu().numpy(), y1.T.cpu().numpy())
    assert_array_equal(y.cpu().numpy(), yt.cpu().numpy())
