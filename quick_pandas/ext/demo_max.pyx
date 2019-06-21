# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True

import numpy as np
import pandas as pd
from quick_pandas.ext.operators cimport OPERATOR_INT, OPERATOR_LONG, OPERATOR_FLOAT, OPERATOR_DOUBLE
from quick_pandas.ext.group_by cimport group_and_transform0



cdef int max_int(int *data, int *indexes, int offset, int end) nogil:
    cdef int res = data[indexes[offset]]
    cdef int i
    for i in range(offset + 1, end):
        if res < data[indexes[i]]:
            res = data[indexes[i]]
    return res

cdef long max_long(long *data, int *indexes, int offset, int end) nogil:
    cdef long res = data[indexes[offset]]
    cdef int i
    for i in range(offset + 1, end):
        if res < data[indexes[i]]:
            res = data[indexes[i]]
    return res

cdef float max_float(float *data, int *indexes, int offset, int end) nogil:
    cdef float res = data[indexes[offset]]
    cdef int i
    for i in range(offset + 1, end):
        if res < data[indexes[i]]:
            res = data[indexes[i]]
    return res

cdef double max_double(double *data, int *indexes, int offset, int end) nogil:
    cdef double res = data[indexes[offset]]
    cdef int i
    for i in range(offset + 1, end):
        if res < data[indexes[i]]:
            res = data[indexes[i]]
    return res


def test():
    df = pd.DataFrame({'A': [1,1,2,2,3,1], 'B': [1,2,3,4,5,6]})
    res = group_and_transform0(df, ['A'], ['B'], 0, max_int, max_long, max_float, max_double, False, False) 
    print(res)


