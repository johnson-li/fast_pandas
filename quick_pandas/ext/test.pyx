from quick_pandas.ext.example cimport func_out, TRANSFORM
import numpy as np


cdef int func_inn(int *data, int length) nogil:
    cdef int i, s = 0
    for i in range(length):
        s += data[i]
    return s


def test():
    cdef int[::1] data = np.random.randint(-100, 100, 100).astype(np.int32)
    cdef TRANSFORM tr = func_inn
    cdef int res = func_out(&data[0], 100, tr)
    print(res)

