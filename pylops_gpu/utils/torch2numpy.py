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

    """
    torchtype = torch.from_numpy(np.ones(1, dtype=numpytype)).dtype
    return torchtype