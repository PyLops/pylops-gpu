# PyLops-GPU
[![AzureDevOps Status](https://dev.azure.com/MRAVA/PyLops/_apis/build/status/equinor.pylops-gpu?branchName=master)](https://dev.azure.com/MRAVA/PyLops/_build/latest?definitionId=2&branchName=master)

:vertical_traffic_light: :vertical_traffic_light: This library is under early development.
Expect things to constantly change until version v1.0.0. :vertical_traffic_light: :vertical_traffic_light:

## Objective
This library is an extension of [PyLops](https://pylops.readthedocs.io/en/latest/)
to run operators on GPUs.

As much as [numpy](http://www.numpy.org) and [scipy](http://www.scipy.org/scipylib/index.html) lie
at the core of the parent project PyLops, PyLops-GPU heavily builds on top of
[PyTorch](http://pytorch.org) and takes advantage of the same optimized
tensor computations used in PyTorch for deep learning using GPUs and CPUs.

Doing so, linear operators can be computed on GPUs.

Here is a simple example showing how a diagonal operator can be created,
applied and inverted using PyLops:
```python
import numpy as np
from pylops import Diagonal

n = 100000
x = np.ones(n)
d = np.arange(n) + 1.

Dop = Diagonal(d)

# y = Dx
y = Dop*x
# xinv = D^-1 y
xinv = Dop / y
```

and similarly using PyLops-gpu:
```python
import numpy as np
import torch
from pylops_gpu.utils.backend import device
from pylops_gpu import Diagonal

dev = device()

n = 100000
x = torch.ones(n, dtype=torch.float64).to(dev)
d = (torch.arange(0, n, dtype=torch.float64) + 1.).to(dev)

Dop = Diagonal(d, device=dev)

# y = Dx
y = Dop*x
# xinv = D^-1 y
xinv = Dop / y
```
