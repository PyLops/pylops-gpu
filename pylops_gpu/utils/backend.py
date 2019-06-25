import torch


def device():
    r"""Automatically identify device to be used with PyTorch

    Returns
    -------
    device : :obj:`str`
        Identified device, ``cpu`` or ``gpu``

    """
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
