from .LinearOperator import LinearOperator, MatrixMult, aslinearoperator
from .basicoperators import Diagonal
from .basicoperators import FirstDerivative

from . import utils
from . import basicoperators
from . import signalprocessing
from . import optimization


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'