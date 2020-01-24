import torch
import numpy as np
from time import time
from pylops_gpu.optimization.leastsquares import NormalEquationsInversion


def _power_iteration(Op, niter=10, tol=1e-5,
                     dtype=torch.float32, device='cpu'):
    """Power iteration algorithm.

    Power iteration algorithm, used to compute the largest eigenvalue and
    corresponding eigenvector. This implementation closely follow that of
    https://en.wikipedia.org/wiki/Power_iteration.
    Parameters
    ----------
    Op : :obj:`pylops_gpu.LinearOperator`
        Operator
    niter : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Update tolerance
    dtype : :obj:`torch.dtype`, optional
        Type of elements in input array.
    device : :obj:`str`, optional
        Device to be used

    Returns
    -------
    maxeig : :obj:`int`
        Largest eigenvalue
    iiter : :obj:`int`
        Effective number of iterations

    """
    # Choose a random vector decrease the chance that vector
    # is orthogonal to the eigenvector
    x = torch.rand(Op.shape[1]).type(dtype).to(device)
    x /= x.norm()
    maxeig_old = torch.tensor(0.).type(dtype).to(device)

    for iiter in range(niter):
        y = Op.matvec(x)
        maxeig = torch.abs(x.dot(y))
        x = y.clone()
        x /= x.norm()

        if torch.abs(maxeig - maxeig_old) < tol * maxeig_old:
            break
        maxeig_old = maxeig.clone()
    return maxeig, iiter + 1


def FISTA(Op, data, niter, eps=0.1, alpha=None, eigsiter=1000, eigstol=0,
          tol=1e-10, returninfo=False, show=False, device='cpu'):
    r"""Fast Iterative Soft Thresholding Algorithm (FISTA).

    Solve an optimization problem with :math:`L1` regularization function given
    the operator ``Op`` and data ``y``. The operator can be real or complex,
    and should ideally be either square :math:`N=M` or underdetermined
    :math:`N<M`.

    Parameters
    ----------
    Op : :obj:`pylops_gpu.LinearOperator`
        Operator to invert
    data : :obj:`torch.tensor`
        Data
    niter : :obj:`int`
        Number of iterations
    eps : :obj:`float`, optional
        Sparsity damping
    alpha : :obj:`float`, optional
        Step size (:math:`\alpha \le 1/\lambda_{max}(\mathbf{Op}^H\mathbf{Op})`
        guarantees convergence. If ``None``, estimated to satisfy the
        condition, otherwise the condition will not be checked)
    eigsiter : :obj:`int`, optional
        Number of iterations for eigenvalue estimation if ``alpha=None``
    eigstol : :obj:`float`, optional
        Tolerance for eigenvalue estimation if ``alpha=None``
    tol : :obj:`float`, optional
        Tolerance. Stop iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    returninfo : :obj:`bool`, optional
        Return info of FISTA solver
    show : :obj:`bool`, optional
        Display iterations log
    device : :obj:`str`, optional
        Device to be used

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    niter : :obj:`int`
        Number of effective iterations
    cost : :obj:`numpy.ndarray`, optional
        History of cost function
    costdata : :obj:`numpy.ndarray`, optional
        History of data fidelity term in the cost function
    costreg : :obj:`numpy.ndarray`, optional
        History of regularizer term in the cost function

    See Also
    --------
    SplitBregman: Split Bregman for mixed L2-L1 norms.

    Notes
    -----
    Solves the following optimization problem for the operator
    :math:`\mathbf{Op}` and the data :math:`\mathbf{d}`:

    .. math::
        J = ||\mathbf{d} - \mathbf{Op} \mathbf{x}||_2^2 +
            \epsilon ||\mathbf{x}||_1

    using the Fast Iterative Soft Thresholding Algorithm (FISTA) [1]_. This is
    a modified version of ISTA solver with improved convergence properties and
    limitied additional computational cost.

    .. [1] Beck, A., and Teboulle, M., “A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems”, SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """
    dtype = data.dtype

    def _softthreshold(x, thresh):
        return torch.max(x.abs() - thresh, torch.zeros_like(x)) * x.sign()

    if show:
        tstart = time()
        print('FISTA optimization\n'
              '-----------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'eps = %10e\ttol = %10e\tniter = %d' % (Op.shape[0],
                                                      Op.shape[1],
                                                      eps, tol, niter))
    # step size
    if alpha is None:
        # compute largest eigenvalues of Op^H * Op
        Op1 = Op.H * Op
        maxeig = torch.abs(_power_iteration(Op1, eigsiter, eigstol,
                                            dtype=Op.dtype, device=device)[0])
        alpha = 1. / maxeig

    # define threshold
    thresh = torch.tensor([eps * alpha * 0.5], device=device, dtype=dtype)

    if show:
        print('alpha = %10e\tthresh = %10e' % (alpha, thresh))
        print('-----------------------------------------------------------\n')
        head1 = '   Itn       x[0]        r2norm     r12norm     xupdate'
        print(head1)

    if returninfo:
        cost = np.zeros(niter+1)
        costdata = np.zeros(niter+1)
        costreg = np.zeros(niter+1)

    # initialize model and cost function
    xinv = torch.zeros(Op.shape[1], dtype=dtype, device=device)
    zinv = xinv.clone()
    t = torch.tensor(1.).to(device)

    # iterate
    for iiter in range(niter):
        xinvold = xinv.clone()

        # compute residual
        resz = data - Op.matvec(zinv)

        # compute gradient
        grad = alpha * Op.rmatvec(resz)

        # update inverted model
        xinv_unthesh = zinv + grad
        xinv = _softthreshold(xinv_unthesh, thresh)

        # update auxiliary coefficients
        told = t.clone()
        t = (1. + torch.sqrt(1. + 4. * t ** 2)) / 2.
        zinv = xinv + ((told - 1.) / t) * (xinv - xinvold)

        # model update
        xupdate = torch.norm(xinv - xinvold)

        if returninfo or show:
            costdata[iiter] = (0.5 * torch.norm(data - Op.matvec(xinv)) ** 2).item()
            costreg[iiter] = (torch.norm(xinv, p=1)).item()
            cost[iiter] = costdata[iiter] + eps * costreg[iiter]

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % 10 == 0:
                msg = '%6g  %12.5e  %10.3e   %9.3e  %10.3e' % \
                      (iiter + 1, xinv[0], costdata[iiter], cost[iiter], xupdate.item())
                print(msg)

        # check tolerance
        if xupdate < tol:
            niter = iiter
            break

    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (niter, time() - tstart))
        print('---------------------------------------------------------\n')
    if returninfo:
        return xinv, niter, cost[:niter], costdata[:niter], costreg[:niter]
    else:
        return xinv, niter


def SplitBregman(Op, RegsL1, data, niter_outer=3, niter_inner=5, RegsL2=None,
                 dataregsL2=None, mu=1., epsRL1s=None, epsRL2s=None, tol=1e-10,
                 tau=1., x0=None, restart=False, show=False, device='cpu',
                 **kwargs_cg):
    r"""Split Bregman for mixed L2-L1 norms.

    Solve an unconstrained system of equations with mixed L2-L1 regularization
    terms given the operator ``Op``, a list of L1 regularization terms
    ``RegsL1``, and an optional list of L2 regularization terms ``RegsL2``.

    Parameters
    ----------
    Op : :obj:`pylops_gpu.LinearOperator`
        Operator to invert
    RegsL1 : :obj:`list`
        L1 regularization operators
    data : :obj:`torch.Tensor`
        Data
    niter_outer : :obj:`int`
        Number of iterations of outer loop
    niter_inner : :obj:`int`
        Number of iterations of inner loop
    RegsL2 : :obj:`list`
        Additional L2 regularization operators
        (if ``None``, L2 regularization is not added to the problem)
    dataregsL2 : :obj:`list`, optional
        L2 Regularization data (must have the same number of elements
        of ``RegsL2`` or equal to ``None`` to use a zero data for every
        regularization operator in ``RegsL2``)
    mu : :obj:`float`, optional
         Data term damping
    epsRL1s : :obj:`list`
         L1 Regularization dampings (must have the same number of elements
         as ``RegsL1``)
    epsRL2s : :obj:`list`
         L2 Regularization dampings (must have the same number of elements
         as ``RegsL2``)
    tol : :obj:`float`, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    tau : :obj:`float`, optional
        Scaling factor in the Bregman update (must be close to 1)
    x0 : :obj:`torch.Tensor`, optional
        Initial guess
    restart : :obj:`bool`, optional
        The unconstrained inverse problem in inner loop is initialized with
        the initial guess (``True``) or with the last estimate (``False``)
    show : :obj:`bool`, optional
        Display iterations log
    device : :obj:`str`, optional
        Device to be used
    **kwargs_cg
        Arbitrary keyword arguments for
        :py:func:`pylops_gpu.optimization.leastsquares.cg` solver

    Returns
    -------
    xinv : :obj:`numpy.ndarray`
        Inverted model
    itn_out : :obj:`int`
        Iteration number of outer loop upon termination

    Notes
    -----
    Solve the following system of unconstrained, regularized equations
    given the operator :math:`\mathbf{Op}` and a set of mixed norm (L2 and L1)
    regularization terms :math:`\mathbf{R_{L2,i}}` and
    :math:`\mathbf{R_{L1,i}}`, respectively:

    .. math::
        J = \mu/2 ||\textbf{d} - \textbf{Op} \textbf{x} |||_2 +
        \sum_i \epsilon_{{R}_{L2,i}} ||\mathbf{d_{{R}_{L2,i}}} -
        \mathbf{R_{L2,i}} \textbf{x} |||_2 +
        \sum_i || \mathbf{R_{L1,i}} \textbf{x} |||_1

    where :math:`\mu` and :math:`\epsilon_{{R}_{L2,i}}` are the damping factors
    used to weight the different terms of the cost function.

    The generalized Split Bergman algorithm is used to solve such cost
    function: the algorithm is composed of a sequence of unconstrained
    inverse problems and Bregman updates. Note that the L1 terms are not
    weighted in the  original cost function but are first converted into
    constraints and then re-inserted in the cost function with Lagrange
    multipliers :math:`\epsilon_{{R}_{L1,i}}`, which effectively act as
    damping factors for those terms. See [1]_ for detailed derivation.

    The :py:func:`scipy.sparse.linalg.lsqr` solver and a fast shrinkage
    algorithm are used within the inner loop to solve the unconstrained
    inverse problem, and the same procedure is repeated ``niter_outer`` times
    until convergence.

    .. [1] Goldstein T. and Osher S., "The Split Bregman Method for
       L1-Regularized Problems", SIAM J. on Scientific Computing, vol. 2(2),
       pp. 323-343. 2008.

    """
    dtype = data.dtype

    def _shrinkage(x, thresh):
        xabs = torch.abs(x)
        return x / (xabs + 1e-10) * torch.max(xabs - thresh, torch.zeros_like(x))

    if show:
        tstart = time()
        print('Split-Bregman optimization\n'
              '---------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'niter_outer = %3d     niter_inner = %3d   tol = %2.2e\n'
              'mu = %2.2e         epsL1 = %s\t  epsL2 = %s     '
              % (Op.shape[0], Op.shape[1],
                 niter_outer, niter_inner, tol,
                 mu, str(epsRL1s), str(epsRL2s)))
        print('---------------------------------------------------------\n')
        head1 = '   Itn          x[0]           r2norm          r12norm'
        print(head1)

    # L1 regularizations
    nregsL1 = len(RegsL1)
    b = [torch.zeros(RegL1.shape[0], dtype=dtype).to(device) for RegL1 in RegsL1]
    d = b.copy()

    # L2 regularizations
    nregsL2 = 0 if RegsL2 is None else len(RegsL2)
    if nregsL2 > 0:
        Regs = RegsL2 + RegsL1
        if dataregsL2 is None:
            dataregsL2 = \
                [torch.zeros(Op.shape[1], dtype=dtype).to(device)] * nregsL2
    else:
        Regs = RegsL1
        dataregsL2 = []

    # Rescale dampings
    epsRs = [np.sqrt(epsRL2s[ireg] / 2) / np.sqrt(mu / 2) for ireg in range(nregsL2)] + \
            [np.sqrt(epsRL1s[ireg] / 2) / np.sqrt(mu / 2) for ireg in range(nregsL1)]
    xinv = x0 if x0 is not None else \
        torch.zeros_like(torch.zeros(Op.shape[1], dtype=dtype).to(device))
    xold = torch.from_numpy(np.inf * np.ones_like(np.zeros(Op.shape[1]))).to(device)

    itn_out = 0
    while (xinv - xold).norm() > tol and itn_out < niter_outer:
        xold = xinv
        for _ in range(niter_inner):
            # Regularized problem
            dataregs = dataregsL2 + [d[ireg] - b[ireg] for ireg in range(nregsL1)]

            xinv = NormalEquationsInversion(Op,
                                            Regs,
                                            data.clone(),
                                            Weight=None,
                                            dataregs=dataregs,
                                            epsRs=epsRs,
                                            x0=x0 if restart else xinv,
                                            returninfo=False,
                                            device=device,
                                            **kwargs_cg)
            # Shrinkage
            d = [_shrinkage(RegsL1[ireg] * xinv + b[ireg], epsRL1s[ireg])
                 for ireg in range(nregsL1)]
        # Bregman update
        b = [b[ireg] + tau * (RegsL1[ireg] * xinv - d[ireg]) for ireg in range(nregsL1)]
        itn_out += 1

        if show:
            costdata = mu / 2. * torch.norm(data - Op.matvec(xinv)).cpu().numpy() ** 2
            costregL2 = 0 if RegsL2 is None else \
                [epsRL2 * np.linalg.norm(dataregL2 - RegL2.matvec(xinv)) ** 2
                 for epsRL2, RegL2, dataregL2 in zip(epsRL2s, RegsL2, dataregsL2)]
            costregL1 = [epsRL1 * torch.norm(RegL1.matvec(xinv), p=1).cpu().numpy()
                         for epsRL1, RegL1 in zip(epsRL1s, RegsL1)]
            cost = costdata + np.sum(np.array(costregL2)) + np.sum(np.array(costregL1))
            msg = '%6g  %12.5e       %10.3e        %9.3e' % \
                  (np.abs(itn_out), xinv[0], costdata, cost)
            print(msg)

    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (itn_out, time() - tstart))
        print('---------------------------------------------------------\n')

    return xinv, itn_out
