.. _installation:

Installation
============

You will need **Python 3.6 or greater** to get started.


Dependencies
------------

Our mandatory dependencies are limited to:

* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org/scipylib/index.html>`_
* `pytorch <https://www.pytorch.org>`_
* `pytorch_complex_tensor <https://github.com/williamFalcon/pytorch-complex-tensor>`_
* `pylops <https://pylops.readthedocs.io/en/latest/#>`_

We advise using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
to ensure that these dependencies are installed via the ``Conda`` package manager.


Step-by-step installation for users
-----------------------------------

Activate your Python environment, and simply type the following command in your terminal
to install the PyPi distribution:

.. code-block:: bash

   >> pip install pylops-gpu

Alternatively, to access the latest source from github:

.. code-block:: bash

   >> pip install https://git@github.com/PyLops/pylops-gpu.git@master

or just clone the repository

.. code-block:: bash

   >> git clone https://github.com/equinor/pylops-gpu.git

or download the zip file from the repository (green button in the top right corner of the
main github repo page) and install PyLops from terminal using the command:

.. code-block:: bash

   >> make install

**Note**: you may see an error if `pytorch-complex-tensor` has not been
previously installed. In that case first run
`pip install pytorch-complex-tensor` and then install pylops-gpu


Step-by-step installation for developers
----------------------------------------
Fork and clone the repository by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/your_name_here/pylops-gpu.git

The first time you clone the repository run the following command:

.. code-block:: bash

   >> make dev-install

If you prefer to build a new Conda enviroment just for PyLops, run the following command:

.. code-block:: bash

   >> make dev-install_conda

To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

    >> make tests

Make sure no tests fail, this guarantees that the installation has been successfull.

If using Conda environment, always remember to activate the conda environment every time you open
a new *bash* shell by typing:

.. code-block:: bash

   >> source activate pylops-gpu