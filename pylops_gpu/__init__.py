from .LinearOperator import LinearOperator
from .basicoperators import MatrixMult, aslinearoperator
from .basicoperators import Diagonal
from .basicoperators import FirstDerivative
from .basicoperators import SecondDerivative
from .basicoperators import Laplacian

from .optimization.cg import cg
from .optimization.leastsquares import NormalEquationsInversion

from . import utils
from . import basicoperators
from . import signalprocessing
from . import optimization


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'