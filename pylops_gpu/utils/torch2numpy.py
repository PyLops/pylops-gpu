import torch
import numpy as np



def numpytype_from_strtype(strtype):
    """Convert str into equivalent numpy type

    Parameters
    ----------
    strtype : :obj:`str`
        String type

    Returns
    -------
    numpytype : :obj:`numpy.dtype`
        Numpy equivalent type

    """
    numpytype = np.dtype(strtype)
    return numpytype


def numpytype_from_torchtype(torchtype):
    """Convert torch type into equivalent numpy type

    Parameters
    ----------
    torchtype : :obj:`torch.dtype`
        Torch type

    Returns
    -------
    numpytype : :obj:`numpy.dtype`
        Numpy equivalent type

    """
    if isinstance(torchtype, torch.dtype):
        numpytype = torch.scalar_tensor(1, dtype=torchtype).numpy().dtype
    else:
        # in case it is already a numpy dtype
        numpytype = torchtype
    return numpytype


def torchtype_from_numpytype(numpytype):
    """Convert torch type into equivalent numpy type

    Parameters
    ----------
    numpytype : :obj:`numpy.dtype`
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
    if isinstance(numpytype, torch.dtype):
        # in case it is already a torch dtype
        torchtype = numpytype
    else:
        torchtype = \
            torch.from_numpy(np.real(np.ones(1, dtype=numpytype_from_strtype(numpytype)))).dtype
    return torchtype