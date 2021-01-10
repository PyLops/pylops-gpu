import torch

from pylops_gpu import LinearOperator
from pylops_gpu.signalprocessing import Convolve1D
from pylops_gpu.utils.torch2numpy import torchtype_from_numpytype


class SecondDerivative(LinearOperator):
    r"""Second derivative.

    Apply second-order centered second derivative.

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
    dtype : :obj:`torch.dtype` or :obj:`np.dtype`, optional
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
    Refer to :class:`pylops.basicoperators.SecondDerivative` for implementation
    details.

    Note that since the Torch implementation is based on a convolution
    with a compact filter :math:`[1., -2., 1.]`, edges are treated
    differently compared to the PyLops equivalent operator.

    """
    def __init__(self, N, dims=None, dir=0, sampling=1., device='cpu',
                 togpu=(False, False), tocpu=(False, False),
                 dtype=torch.float32):
        # convert dtype to torch.dtype
        dtype = torchtype_from_numpytype(dtype)

        h = torch.torch.tensor([1., -2, 1.],
                               dtype=dtype).to(device) / sampling**2
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu
        self.shape = (N, N)
        self.dtype = dtype
        self.explicit = False
        self.Op = Convolve1D(N, h, offset=1, dims=dims, dir=dir,
                             zero_edges=True, device=device,
                             togpu=togpu, tocpu=tocpu, dtype=dtype)
