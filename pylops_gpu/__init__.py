from .LinearOperator import LinearOperator
from .TorchOperator import TorchOperator
from .basicoperators import MatrixMult, aslinearoperator
from .basicoperators import Diagonal
from .basicoperators import Identity
from .basicoperators import Restriction
from .basicoperators import VStack
from .basicoperators import FirstDerivative
from .basicoperators import SecondDerivative
from .basicoperators import Laplacian

from .optimization.cg import cg, cgls
from .optimization.leastsquares import NormalEquationsInversion

from .avo.poststack import PoststackLinearModelling

from . import avo
from . import basicoperators
from . import signalprocessing
from . import optimization
from . import utils


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'