import torch

from pylops_gpu import LinearOperator
from pylops_gpu.signalprocessing import Convolve1D


"""
def FirstDerivative(N, dims=None, dir=0, sampling=1., device='cpu',
                    togpu=(False, False), tocpu=(False, False),
                    dtype=torch.float32):

    h = torch.torch.tensor([0.5, 0, -0.5], dtype=dtype).to(device) / sampling
    dop = Convolve1D(N, h, offset=1, dims=dims, dir=dir, device=device,
                     togpu=togpu, tocpu=tocpu, dtype=dtype)
    return dop
"""

class FirstDerivative(LinearOperator):
    r"""First derivative.

    Apply second-order centered first derivative.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    dtype : :obj:`torch.dtype`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.FirstDerivative` for implementation
    details.

    Note that since the Torch implementation is based on a convolution
    with a compact filter :math:`[0.5, 0., -0.5]`, edges are treated
    differently compared to the PyLops equivalent operator.

    """
    def __init__(self, N, dims=None, dir=0, sampling=1., device='cpu',
                 togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float32):
        h = torch.torch.tensor([0.5, 0, -0.5],
                               dtype=dtype).to(device) / sampling
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.shape = (N, N)
        self.dtype = dtype
        self.explicit = False
        self.Op = Convolve1D(N, h, offset=1, dims=dims, dir=dir,
                             zero_edges=True, device=device,
                             togpu=togpu, tocpu=tocpu, dtype=dtype)
