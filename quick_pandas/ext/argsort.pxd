# cython: language_level=3

import numpy as np
from typing import List

cdef int C_ARRAY_TYPE_INT64
cdef int C_ARRAY_TYPE_FLOAT64
cdef int C_ARRAY_TYPE_STRING
cdef int[::1] unwrap_arrays(arrays: List[np.ndarray], unsigned char **c_arrays)
cdef void radix_argsort(unsigned char **arrays, int[::1] dtypes, int arrays_length,
                        int[::1] indexes, int array_index, int array_offset, int array_length) nogil
cdef int compare(unsigned char **arrays, int[::1] dtypes,
                 int array_length, int array_index, int l, int r) nogil
