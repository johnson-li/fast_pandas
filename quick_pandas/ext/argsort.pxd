# cython: language_level=3

import numpy as np
from typing import List


cdef int C_ARRAY_TYPE_INT64
cdef int C_ARRAY_TYPE_INT32
cdef int C_ARRAY_TYPE_FLOAT64
cdef int C_ARRAY_TYPE_FLOAT32
cdef int C_ARRAY_TYPE_STRING
cdef int C_STRING_TYPE_OFFSET_BITS
cdef int C_STRING_TYPE_OFFSET_MASK
cdef int* unwrap_arrays(arrays: List[np.ndarray], unsigned char **c_arrays)
cdef void radix_argsort(unsigned char **arrays, int *dtypes, int arrays_length,
                        int *indexes, int array_index, int array_offset, int array_length,
                        int *ranges, int *range_length) nogil
cdef void radix_argsort_groups(unsigned char **arrays, int *dtypes, int arrays_length,
                               int *indexes, int array_index, int array_offset, int array_length,
                               int *ranges, int *range_length) nogil
cdef int compare(unsigned char **arrays, int *dtypes,
                 int array_length, int array_index, int l, int r) nogil
