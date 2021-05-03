.. _changlog:

Changelog
=========


Version 0.0.1
-------------

*Released on: 03/05/2021*

* Added :py:func:`pylops_gpu.optimization.sparsity.FISTA` and
  :py:func:`pylops_gpu.optimization.sparsity.SplitBregman` solvers
* Modified :py:class:`pylops_gpu.TorchOperator` to work with cupy arrays
* Modified :py:func:`pylops_gpu.avo.poststack._PoststackLinearModelling` to use
  the code written in pylops library whilst still dealing with torch arrays
* Allowed passing numpy dtypes to operators (automatic conversion
  to torch types)

Version 0.0.0
-------------

*Released on: 12/01/2020*

* First official release.
