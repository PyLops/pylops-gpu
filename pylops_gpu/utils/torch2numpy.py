import torch
import numpy as np


def numpytype_from_torchtype(torchtype):
    """Convert torch type into equivalent numpy type

    Parameters
    ----------
    torchtype : :obj:`torch.dtype`
        Torch type

    Returns
    -------
    numpytype : :obj:`torch.dtype`
        Numpy equivalent type

    """
    numpytype = torch.scalar_tensor(1, dtype=torchtype).numpy().dtype
    return numpytype


def torchtype_from_numpytype(numpytype):
    """Convert torch type into equivalent numpy type

    Parameters
    ----------
    numpytype : :obj:`torch.dtype`
        Numpy type

    Returns
    -------
    torchtype : :obj:`torch.dtype`
        Torch equivalent type

    Notes
    -----
    Given limitations of torch to handle complex numbers, complex numpy types
    are casted into equivalent real types and the equivalent torch type is
    returned.

    """
    torchtype = torch.from_numpy(np.real(np.ones(1, dtype=numpytype))).dtype
    return torchtype