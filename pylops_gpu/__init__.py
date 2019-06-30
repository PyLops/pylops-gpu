from .LinearOperator import LinearOperator, MatrixMult, aslinearoperator
from .basicoperators import Diagonal

from . import utils
from . import basicoperators
from . import optimization


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'