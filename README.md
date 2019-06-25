# PyLops-GPU

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

n = 10
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
import pylops_gpu
from pylops_gpu import Diagonal
from scipy.sparse.linalg import lsqr

dev = device()

n = 10
x = torch.ones(n, dtype=torch.float64).to(dev)
d = (torch.arange(0, n, dtype=torch.float64) + 1.).to(dev)

Dop = Diagonal(d, device=dev, tonumpy=False)

# y = Dx
y = Dop*x
# xinv = D^-1 y
xinv = Dop / y
```
