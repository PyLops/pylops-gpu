import torch
from scipy.sparse.linalg.interface import \
    _ProductLinearOperator, _ScaledLinearOperator
from pylops import LinearOperator as pLinearOperator

class LinearOperator(pLinearOperator):
    """Common interface for performing matrix-vector products.

    This class is an overload of the
    :py:class:`pylops.LinearOperator` class. It adds
    functionalities for operators on GPUs; specifically, it allows users
    specifying when to move model and data from the host to the device and
    viceversa.

    Compared to its equivalent PyLops class :class:`pylops.LinearOperator`, it
    requires input model and data to be :obj:`torch.Tensor` objects.

    .. note:: End users of PyLops should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyLops-gpu. Find more details regarding
      implementation of new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)

    """
    def __init__(self, Op=None, explicit=False, device='cpu',
                 togpu=(False, False), tocpu=(False, False)):
        super().__init__(Op=Op, explicit=explicit)
        self.device = device
        self.togpu = togpu
        self.tocpu = tocpu

    def matvec(self, x):
        # convert x to torch.Tensor
        if not isinstance(x, torch.Tensor):
            _tonumpy = True
            x = torch.from_numpy(x)
        else:
            _tonumpy = False
        # matvec, possibly moving x to gpu and y back to cpu
        if self.device != 'cpu' and self.togpu[0]:
            x = x.to(self.device)
        if self.Op is None:
            y = self._matvec(x)
        else:
            y = self.Op._matvec(x)
        if self.device != 'cpu' and self.tocpu[0]:
            y = y.to('cpu')
        # convert y to numpy when input was numpy
        if _tonumpy:
            y = y.numpy()
        return y

    def rmatvec(self, x):
        # convert x to torch.Tensor
        if not isinstance(x, torch.Tensor):
            _tonumpy = True
            x = torch.from_numpy(x)
        else:
            _tonumpy = False
        # rmatvec, possibly moving x to gpu and y back to cpu
        if self.device != 'cpu' and self.togpu[1]:
            x = x.to(self.device)
        if self.Op is None:
            y = self._rmatvec(x)
        else:
            y = self.Op._rmatvec(x)
        if self.device != 'cpu' and self.tocpu[1]:
            y = y.to('cpu')
        # convert y to numpy when input was numpy
        if _tonumpy:
            y = y.numpy()
        return y

    def __mul__(self, x):
        return self.matvec(x)