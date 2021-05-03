# 0.0.1
* Added ``pylops_gpu.optimization.sparsity.FISTA`` and
  ``pylops_gpu.optimization.sparsity.SplitBregman`` solvers
* Modified ``pylops_gpu.TorchOperator`` to work with cupy arrays
* Modified ``pylops_gpu.avo.poststack._PoststackLinearModelling`` to use 
  the code written in pylops library whilst still dealing with torch arrays
* Allowed passing numpy dtypes to operators (automatic conversion 
  to torch types)

# 0.0.0
* First official release.
