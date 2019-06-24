# cython: language_level=3

cdef int sum_int(int *data, int *indexes, int start, int end) nogil:
    cdef int res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]] 
    return res

cdef long sum_long(long *data, int *indexes, int start, int end) nogil:
    cdef long res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]] 
    return res

cdef float sum_float(float *data, int *indexes, int start, int end) nogil:
    cdef float res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]] 
    return res

cdef double sum_double(double *data, int *indexes, int start, int end) nogil:
    cdef double res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]] 
    return res

cdef int mean_int(int *data, int *indexes, int start, int end) nogil:
    cdef int res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]]
    return <int>(res / (end - start))

cdef long mean_long(long *data, int *indexes, int start, int end) nogil:
    cdef long res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]]
    return <long>(res / (end - start))


cdef float mean_float(float *data, int *indexes, int start, int end) nogil:
    cdef float res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]]
    return <float>(res / (end - start))

cdef double mean_double(double *data, int *indexes, int start, int end) nogil:
    cdef double res = 0
    cdef int i
    for i in range(start, end):
        res += data[indexes[i]]
    return <double>(res / (end - start))

