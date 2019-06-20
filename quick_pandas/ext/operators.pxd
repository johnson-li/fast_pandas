# cython: language_level=3


ctypedef int (*OPERATOR_INT)(int *data, int *indexes, int offset, int length) nogil
ctypedef long (*OPERATOR_LONG)(long *data, int *indexes, int offset, int length) nogil
ctypedef float (*OPERATOR_FLOAT)(float *data, int *indexes, int offset, int length) nogil
ctypedef double (*OPERATOR_DOUBLE)(double *data, int *indexes, int offset, int length) nogil


cdef int sum_int(int *data, int *indexes, int offset, int length) nogil
cdef long sum_long(long *data, int *indexes, int offset, int length) nogil
cdef float sum_float(float *data, int *indexes, int offset, int length) nogil
cdef double sum_double(double *data, int *indexes, int offset, int length) nogil

cdef int mean_int(int *data, int *indexes, int offset, int length) nogil
cdef long mean_long(long *data, int *indexes, int offset, int length) nogil
cdef float mean_float(float *data, int *indexes, int offset, int length) nogil
cdef double mean_double(double *data, int *indexes, int offset, int length) nogil

