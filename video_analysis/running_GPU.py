# from numba import jit, cuda
# import numba
import numpy as np
import timeit
import cupy as cp

def creat_empty_arraies(N,run_cp=True):
    if run_cp:
        x = cp.random.randn(N).astype(cp.float32)
        y = cp.random.randn(N).astype(cp.float32)
        z = x+y
    else:
        x = np.random.randn(N).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)
        z = x+y

start = timeit.default_timer()
creat_empty_arraies(1000000)
print('Time_cupy: ', timeit.default_timer() - start)
start = timeit.default_timer()
creat_empty_arraies(1000000,run_cp=False)
print('Time_numpy: ', timeit.default_timer() - start)
# import numpy as np
# from numba import cuda
# import numba
#
#
# @numba.jit(target='cuda')
# def function(ar):
#     for i in range(3):
#         ar[i] = (1, 2, 3)
#     return ar
#
# print(function(np.zeros((3, 3))))