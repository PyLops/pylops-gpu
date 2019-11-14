import logging
import torch
import numpy as np

from scipy.sparse import csc_matrix
from pylops import MatrixMult, FirstDerivative
from pylops.utils.signalprocessing import convmtx, nonstationary_convmtx
from pylops.signalprocessing import Convolve1D
#from pylops.avo.poststack import _PoststackLinearModelling

from pylops_gpu.utils import dottest as Dottest
from pylops_gpu import MatrixMult as gMatrixMult
from pylops_gpu import FirstDerivative as gFirstDerivative
from pylops_gpu import SecondDerivative as gSecondDerivative
from pylops_gpu import Laplacian as gLaplacian
from pylops_gpu.signalprocessing import Convolve1D as gConvolve1D
from pylops_gpu.optimization.cg import cg
from pylops_gpu.optimization.leastsquares import RegularizedInversion

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _PoststackLinearModelling(wav, nt0, spatdims=None, explicit=False,
                              sparse=False, _MatrixMult=MatrixMult,
                              _Convolve1D=Convolve1D,
                              _FirstDerivative=FirstDerivative,
                              args_MatrixMult={}, args_Convolve1D={},
                              args_FirstDerivative={}):


    """Post-stack linearized seismic modelling operator.

    Used to be able to provide operators from different libraries to
    PoststackLinearModelling. It operates in the same way as public method
    (PoststackLinearModelling) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    if len(wav.shape) == 2 and wav.shape[0] != nt0:
        raise ValueError('Provide 1d wavelet or 2d wavelet composed of nt0 '
                         'wavelets')

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
        if len(wav.shape) == 1:
            C = convmtx(wav, nt0)[:, len(wav) // 2:-len(wav) // 2 + 1]
        else:
            C = nonstationary_convmtx(wav, nt0, hc=wav.shape[1] // 2,
                                      pad=(nt0, nt0))
        # Combine operators
        M = np.dot(C, D)
        if sparse:
            M = csc_matrix(M)
        Pop = _MatrixMult(M, dims=spatdims, **args_MatrixMult)
    else:
        # Create wavelet operator
        if len(wav.shape) == 1:
            Cop = _Convolve1D(np.prod(np.array(dims)), h=wav,
                              offset=len(wav) // 2, dir=0, dims=dims,
                              **args_Convolve1D)
        else:
            Cop = _MatrixMult(nonstationary_convmtx(wav, nt0,
                                                    hc=wav.shape[1] // 2,
                                                    pad=(nt0, nt0)),
                              dims=spatdims, **args_MatrixMult)
        # Create derivative operator
        Dop = _FirstDerivative(np.prod(np.array(dims)), dims=dims,
                               dir=0, sampling=1., **args_FirstDerivative)
        Pop = Cop * Dop
    return Pop


def PoststackLinearModelling(wav, nt0, spatdims=None, explicit=False,
                             device='cpu', togpu=(False, False),
                             tocpu=(False, False)):
    r"""Post-stack linearized seismic modelling operator.

    Create operator to be applied to an elastic parameter trace (or stack of
    traces) for generation of band-limited seismic post-stack data. The input
    model and data have shape :math:`[n_{t0} (\times n_x \times n_y)]`.

    Parameters
    ----------
    wav : :obj:`torch.Tensor` or :obj:`np.ndarray`
        Wavelet in time domain (must have odd number of elements
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
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)

    Returns
    -------
    Pop : :obj:`LinearOperator`
        post-stack modelling operator.

    Notes
    -----
    Refer to :class:`pylops.avo.poststack.PoststackLinearModelling` for
    implementation details.

    """
    if not isinstance(wav, torch.Tensor) and not explicit:
        wav = torch.from_numpy(wav).to(device)
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


def PoststackInversion(data, wav, m0=None, explicit=False,
                       simultaneous=False, epsI=None, epsR=None,
                       dottest=False, epsRL1=None,
                       device='cpu', togpu=(False, False),
                       tocpu=(False, False), **kwargs_solver):
    r"""Post-stack linearized seismic inversion.

    Invert post-stack seismic operator to retrieve an acoustic
    impedance profile from band-limited seismic post-stack data.
    Depending on the choice of input parameters, inversion can be
    trace-by-trace with explicit operator or global with either
    explicit or linear operator.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Band-limited seismic post-stack data of size
        :math:`[n_{t0} (\times n_x \times n_y)]`
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must have odd number of elements
        and centered to zero). If 1d, assume stationary wavelet for the entire
        time axis. If 2d of size :math:`[n_{t0} \times n_h]` use as
        non-stationary wavelet
    m0 : :obj:`np.ndarray`, optional
        Background model of size :math:`[n_{t0} (\times n_x \times n_y)]`
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preferred for large data)
        or a ``MatrixMult`` linear operator with dense matrix
        (``True``, preferred for small data)
    simultaneous : :obj:`bool`, optional
        Simultaneously invert entire data (``True``) or invert
        trace-by-trace (``False``) when using ``explicit`` operator
        (note that the entire data is always inverted when working
        with linear operator)
    epsI : :obj:`float`, optional
        Damping factor for Tikhonov regularization term
    epsR : :obj:`float`, optional
        Damping factor for additional Laplacian regularization term
    dottest : :obj:`bool`, optional
        Apply dot-test
    epsRL1 : :obj:`float`, optional
        Damping factor for additional blockiness regularization term
    device : :obj:`str`, optional
        Device to be used
    togpu : :obj:`tuple`, optional
        Move model and data from cpu to gpu prior to applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    tocpu : :obj:`tuple`, optional
        Move data and model from gpu to cpu after applying ``matvec`` and
        ``rmatvec``, respectively (only when ``device='gpu'``)
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.linalg.lstsq`
        solver (if ``explicit=True`` and  ``epsR=None``)
        or :py:func:`scipy.sparse.linalg.lsqr` solver (if ``explicit=False``
        and/or ``epsR`` is not ``None``)

    Returns
    -------
    minv : :obj:`np.ndarray`
        Inverted model of size :math:`[n_{t0} (\times n_x \times n_y)]`
    datar : :obj:`np.ndarray`
        Residual data (i.e., data - background data) of
        size :math:`[n_{t0} (\times n_x \times n_y)]`

    Notes
    -----
    Refer to :class:`pylops.avo.poststack.PoststackInversion` for
    implementation details.

    """
    # check if background model and data have same shape
    if m0 is not None and data.shape != m0.shape:
        raise ValueError('data and m0 must have same shape')

    # find out dimensions
    if len(data.shape) == 1:
        dims = 1
        nt0 = data.shape[0]
        nspat = None
        nspatprod = nx = 1
    elif len(data.shape) == 2:
        dims = 2
        nt0, nx = data.shape
        nspat = (nx,)
        nspatprod = nx
    else:
        dims = 3
        nt0, nx, ny = data.shape
        nspat = (nx, ny)
        nspatprod = nx * ny
        data = data.reshape(nt0, nspatprod)

    # create operator
    PPop = PoststackLinearModelling(wav, nt0=nt0, spatdims=nspat,
                                    explicit=explicit, tocpu=tocpu,
                                    togpu=togpu, device=device)
    if dottest:
        Dottest(PPop, nt0 * nspatprod, nt0 * nspatprod,
                raiseerror=True, verb=True)

    # create and remove background data from original data
    datar = data.flatten() if m0 is None else \
        data.flatten() - PPop * m0.flatten()
    # invert model
    if epsR is None:
        # inversion without spatial regularization
        if explicit:
            if epsI is None and not simultaneous:
                # solve unregularized equations indipendently trace-by-trace
                minv = torch.solve(datar.reshape(nt0, nspatprod),
                                   PPop.A.reshape(nt0, nt0) +
                                   1e-3 * torch.eye(nt0, dtype=torch.float32)).solution
            elif epsI is None and simultaneous:
                # solve unregularized equations simultaneously
                minv = cg(PPop.H * PPop, PPop.H * datar, **kwargs_solver)[0]
            elif epsI is not None:
                # create regularized normal equations
                PP = torch.matmul(PPop.A.t(), PPop.A) + \
                     epsI * torch.eye(nt0, dtype=torch.float32)
                datarn = torch.matmul(PPop.A.t(), datar.reshape(nt0, nspatprod))
                if not simultaneous:
                    # solve regularized normal eqs. trace-by-trace
                    minv = torch.solve(datarn.reshape(nt0, nspatprod),
                                       PP).solution
                else:
                    # solve regularized normal equations simultaneously
                    PPop_reg = gMatrixMult(PP, dims=nspatprod, device=device,
                                           togpu=togpu, tocpu=tocpu)
                    minv = cg(PPop_reg.H * PPop_reg,
                              PPop_reg.H * datar.flatten(),
                              **kwargs_solver)[0]
            else:
                # create regularized normal eqs. and solve them simultaneously
                PP = np.dot(PPop.A.T, PPop.A) + epsI * np.eye(nt0)
                datarn = PPop.A.T * datar.reshape(nt0, nspatprod)
                PPop_reg = gMatrixMult(PP, dims=nspatprod, device=device,
                                       togpu=togpu, tocpu=tocpu)
                minv = torch.solve(datarn.reshape(nt0, nspatprod),
                                   PPop_reg.A).solution
        else:
            # solve unregularized normal equations simultaneously with lop
            minv = cg(PPop.H * PPop, PPop.H * datar, **kwargs_solver)[0]
    else:
        if epsRL1 is None:
            # L2 inversion with spatial regularization
            if dims == 1:
                Regop = gSecondDerivative(nt0, device=device,
                                          togpu=togpu, tocpu=tocpu,
                                          dtype=PPop.dtype)
            elif dims == 2:
                Regop = gLaplacian((nt0, nx), device=device,
                                   togpu=togpu, tocpu=tocpu,
                                   dtype=PPop.dtype)
            else:
                Regop = gLaplacian((nt0, nx, ny), dirs=(1, 2), device=device,
                                   togpu=togpu, tocpu=tocpu, dtype=PPop.dtype)

            minv = RegularizedInversion(PPop, [Regop], data.flatten(),
                                        x0=None if m0 is None else m0.flatten(),
                                        epsRs=[epsR], **kwargs_solver)[0]
        else:
            # Blockiness-promoting inversion with spatial regularization
            raise NotImplementedError('SplitBregman not available...')

    # compute residual
    if epsR is None:
        datar -= PPop * minv.flatten()
    else:
        datar = data.flatten() - PPop * minv.flatten()

    # reshape inverted model and residual data
    if dims == 1:
        minv = minv.squeeze()
        datar = datar.squeeze()
    elif dims == 2:
        minv = minv.reshape(nt0, nx)
        datar = datar.reshape(nt0, nx)
    else:
        minv = minv.reshape(nt0, nx, ny)
        datar = datar.reshape(nt0, nx, ny)

    if m0 is not None and epsR is None:
        minv = minv + m0

    return minv, datar
