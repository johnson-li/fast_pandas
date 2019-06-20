# cython: language_level=3

ctypedef int (*TRANSFORM)(int *data, int length) nogil

cdef int func_out(int *data, int length, TRANSFORM f) nogil

