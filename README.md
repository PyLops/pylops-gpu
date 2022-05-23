![PyLops-gpu](https://github.com/PyLops/pylops-gpu/blob/master/docs/source/_static/g-pylops_b.png)

[![PyPI version](https://badge.fury.io/py/pylops-gpu.svg)](https://badge.fury.io/py/pylops-gpu)
[![Build Status](https://travis-ci.com/PyLops/pylops-gpu.svg?branch=master)](https://travis-ci.com/PyLops/pylops-gpu)
[![AzureDevOps Status](https://dev.azure.com/matteoravasi/PyLops/_apis/build/status/PyLops.pylops-gpu?branchName=master)](https://dev.azure.com/matteoravasi/PyLops/_build/latest?definitionId=7&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/pylops-gpu/badge/?version=latest)](https://pylops-gpu.readthedocs.io/en/latest/?badge=latest)
[![OS-support](https://img.shields.io/badge/OS-linux,osx-850A8B.svg)](https://github.com/PyLops/pylops-gpu)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)


:vertical_traffic_light: :vertical_traffic_light: This library is unmantained!
If interested in running PyLops on GPUs please use the cupy backend from the PyLops main library :vertical_traffic_light: :vertical_traffic_light:

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

n = int(1e6)
x = np.ones(n)
d = np.arange(n) + 1.

Dop = Diagonal(d)

# y = Dx
y = Dop*x
```

and similarly using PyLops-gpu:
```python
import numpy as np
import torch
from pylops_gpu.utils.backend import device
from pylops_gpu import Diagonal

dev = device()

n = int(1e6)
x = torch.ones(n, dtype=torch.float64).to(dev)
d = (torch.arange(0, n, dtype=torch.float64) + 1.).to(dev)

Dop = Diagonal(d, device=dev)

# y = Dx
y = Dop*x
```

Running these two snippets of code in Google Colab with GPU enabled gives a 50+
speed up for the forward pass.

As a by-product of implementing PyLops linear operators in PyTorch, we can easily
chain our operators with any nonlinear mathematical operation (e.g., log, sin, tan, pow, ...)
as well as with operators from the ``torch.nn`` submodule and obtain *Automatic
Differentiation* (AD) for the entire chain. Since the gradient of a linear
operator is simply its *adjoint*, we have implemented a single class,
`pylops_gpu.TorchOperator`, which can wrap any linear operator
from PyLops and PyLops-gpu libraries and return a `torch.autograd.Function` object.


## Project structure
This repository is organized as follows:
* **pylops_gpu**: python library containing various GPU-powered linear operators and auxiliary routines
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       sphinx documentation
* **examples**:   set of python script examples for each linear operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery

## Getting started

You need **Python 3.5 or greater**.

#### From PyPi

If you want to use PyLops-gpu within your codes,
install it in your Python-gpu environment by typing the following command in your terminal:

```
pip install pylops-gpu
```

Open a python terminal and type:

```
import pylops_gpu
```

If you do not see any error, you should be good to go, enjoy!

**Note**: you may see an error if `pytorch-complex-tensor` has not been
previously installed. In that case first run 
`pip install pytorch-complex-tensor` and then install pylops-gpu

#### From Github

You can also directly install from the master node

```
pip install git+https://git@github.com/PyLops/pylops-gpu.git@master
```

## Contributing
*Feel like contributing to the project? Adding new operators or tutorial?*

Follow the instructions from [PyLops official documentation](https://pylops.readthedocs.io/en/latest/contributing.html).

## Documentation
The official documentation of PyLops-gpu is available [here](https://pylops-gpu.readthedocs.io/).

Visit this page to get started learning about different operators and their applications as well as how to
create new operators yourself and make it to the ``Contributors`` list.

Moreover, if you have installed PyLops using the *developer environment* you can also build the documentation locally by
typing the following command:
```
make doc
```
Once the documentation is created, you can make any change to the source code and rebuild the documentation by
simply typing
```
make docupdate
```
Note that if a new example or tutorial is created (and if any change is made to a previously available example or tutorial)
you are required to rebuild the entire documentation before your changes will be visible.


## History
PyLops-GPU was initially written and it is currently maintained by [Equinor](https://www.equinor.com).
It is an extension of [PyLops](https://pylops.readthedocs.io/en/latest/) for large-scale optimization with
*GPU-driven* linear operators on that can be tailored to our needs, and as contribution to the free software community.


## Contributors
* Matteo Ravasi, mrava87
* Francesco Picetti, fpicetti