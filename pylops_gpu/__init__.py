from .LinearOperator import LinearOperator

from .basicoperators import Diagonal

from . import basicoperators
from . import utils


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'