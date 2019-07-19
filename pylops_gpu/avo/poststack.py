import logging
import torch
import numpy as np
from scipy.sparse import csc_matrix

from pylops import MatrixMult, FirstDerivative
from pylops.utils.signalprocessing import convmtx
from pylops.signalprocessing import Convolve1D
from pylops_gpu import MatrixMult as gMatrixMult
from pylops_gpu import FirstDerivative as gFirstDerivative
from pylops_gpu.signalprocessing import Convolve1D as gConvolve1D

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _PoststackLinearModelling(wav, nt0, spatdims=None, explicit=False,
                              sparse=False, _MatrixMult=MatrixMult,
                              _Convolve1D=Convolve1D,
                              _FirstDerivative=FirstDerivative,
                              args_MatrixMult={}, args_Convolve1D={},
                              args_FirstDerivative={}):
    """Post-stack linearized seismic modelling operator.

    Used to be able to provide operators from different libraries to
    PoststackLinearModelling.

    """
    # organize dimensions
    if spatdims is None:
        dims = (nt0,)
        spatdims = None
    elif isinstance(spatdims, int):
        dims = (nt0, spatdims)
        spatdims = (spatdims,)
    else:
        dims = (nt0,) + spatdims

    if explicit:
        # Create derivative operator
        D = np.diag(0.5 * np.ones(nt0 - 1), k=1) - \
            np.diag(0.5 * np.ones(nt0 - 1), -1)
        D[0] = D[-1] = 0

        # Create wavelet operator
        C = convmtx(wav, nt0)[:, len(wav) // 2:-len(wav) // 2 + 1]

        # Combine operators
        M = np.dot(C, D)
        if sparse:
            M = csc_matrix(M)
        Pop = _MatrixMult(M, dims=spatdims, **args_MatrixMult)
    else:
        # Create wavelet operator
        Cop = _Convolve1D(np.prod(np.array(dims)), h=wav,
                          offset=len(wav) // 2, dir=0, dims=dims,
                          **args_Convolve1D)
        # Create derivative operator
        Dop = _FirstDerivative(np.prod(np.array(dims)), dims=dims,
                               dir=0, sampling=1., **args_FirstDerivative)
        Pop = Cop * Dop
    return Pop


def PoststackLinearModelling(wav, nt0, spatdims=None, explicit=False,
                             device='cpu', togpu=(False, False),
                             tocpu=(False, False)):
    r"""Post-stack linearized seismic modelling operator.

    Create operator to be applied to an acoustic impedance trace (or stack of
    traces) for generation of band-limited seismic post-stack data. The input
    model and data have shape :math:`[n_{t0} (\times n_x \times n_y)]`.

    Parameters
    ----------
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must had odd number of elements
        and centered to zero)
    nt0 : :obj:`int`
        Number of samples along time axis
    spatdims : :obj:`int` or :obj:`tuple`, optional
        Number of samples along spatial axis (or axes)
        (``None`` if only one dimension is available)
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix (``True``,
        preferred for small data)

    Returns
    -------
    Pop : :obj:`LinearOperator`
        post-stack modelling operator.

    Notes
    -----
    Refer to :class:`pylops.avo.poststack.PoststackLinearModelling` for
    implementation details.

    """
    if not isinstance(wav, torch.Tensor):
        wav = torch.from_numpy(wav)
    return _PoststackLinearModelling(wav, nt0, spatdims=spatdims,
                                     explicit=explicit, sparse=False,
                                     _MatrixMult=gMatrixMult,
                                     _Convolve1D=gConvolve1D,
                                     _FirstDerivative=gFirstDerivative,
                                     args_MatrixMult={'device':device,
                                                      'togpu':(togpu[0], togpu[1]),
                                                      'tocpu':(tocpu[0], tocpu[1])},
                                     args_Convolve1D={'device':device,
                                                      'togpu':(False, togpu[1]),
                                                      'tocpu':(tocpu[0], False)},
                                     args_FirstDerivative={'device':device,
                                                           'togpu':(togpu[0], False),
                                                           'tocpu':(False, tocpu[1])})
