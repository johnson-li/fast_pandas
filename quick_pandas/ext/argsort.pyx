# tag: openmp
# cython: profile=False
# cython: language_level=3
# cython: linetrace=False
# cython: binding=False
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

from typing import List
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport isnan
from cython.parallel import prange
import numpy as np
from quick_pandas.dtypes import ARRAY_TYPE_INT32, ARRAY_TYPE_INT64, ARRAY_TYPE_FLOAT32, ARRAY_TYPE_FLOAT64, ARRAY_TYPE_STRING, \
    STRING_TYPE_OFFSET_BITS, STRING_TYPE_OFFSET_MASK, convert_to_uint8

cdef int C_ARRAY_TYPE_INT64 = ARRAY_TYPE_INT64
cdef int C_ARRAY_TYPE_INT32 = ARRAY_TYPE_INT32
cdef int C_ARRAY_TYPE_FLOAT64 = ARRAY_TYPE_FLOAT64
cdef int C_ARRAY_TYPE_FLOAT32 = ARRAY_TYPE_FLOAT32
cdef int C_ARRAY_TYPE_STRING = ARRAY_TYPE_STRING
cdef int C_STRING_TYPE_OFFSET_BITS = STRING_TYPE_OFFSET_BITS
cdef int C_STRING_TYPE_OFFSET_MASK = STRING_TYPE_OFFSET_MASK

ctypedef unsigned long ulong
ctypedef unsigned char uchar
cdef int INSERTION_SORT_LIMIT = 64
cdef int RADIX_BITS = 8


cdef int compare0(const unsigned char* array, int dtype, int l, int r) nogil:
    cdef double l_double, r_double
    cdef long l_long, r_long
    cdef int l_int = 0, r_int = 0, str_length, i
    cdef int* array_int
    cdef long* array_long
    cdef double* array_double
    if dtype == C_ARRAY_TYPE_INT64:
        array_long = <long*> array
        l_long = array_long[l]
        r_long = array_long[r]
        if l_long < r_long:
            return -1
        if l_long > r_long:
            return 1
        return 0
    elif dtype == C_ARRAY_TYPE_FLOAT64:
        array_double = <double*> array
        l_double = array_double[l]
        r_double = array_double[r]
        if isnan(l_double):
            if isnan(r_double):
                return 0
            return 1
        if isnan(r_double):
            return -1
        if l_double < r_double:
            return -1
        if l_double > r_double:
            return 1
        return 0
    elif dtype & C_STRING_TYPE_OFFSET_MASK == C_ARRAY_TYPE_STRING:
        str_length = dtype >> C_STRING_TYPE_OFFSET_BITS
        array_int = <int*> array
        for i in range(str_length):
            l_int = array_int[l * str_length + i]
            r_int = array_int[r * str_length + i]
            if l_int < r_int:
                return -1
            if l_int > r_int:
                return 1
            if l_int == 0:
                return 0
        return 0
    return -2


cdef int compare(unsigned char **arrays, int *dtypes,
                 int array_length, int array_index, int l, int r) nogil:
    cdef int res = 0
    cdef int dtype = 0
    cdef unsigned char *array
    while array_index < array_length:
        dtype = dtypes[array_index]
        array = arrays[array_index]
        res = compare0(array, dtype, l, r)
        if res != 0:
            return res
        array_index += 1
    return 0


cdef inline void add_group(int *ranges, int *range_length, int offset) nogil:
    if ranges != NULL:
        ranges[range_length[0]] = offset
        range_length[0] += 1


cdef void insertion_argsort(unsigned char **arrays, int *dtypes, int arrays_length,
                            int *indexes, int array_index, int array_offset, int array_length,
                            int *ranges, int *range_length) nogil:
    cdef int i, j, tmp, cmp
    for i in range(array_offset, array_offset + array_length):
        tmp = indexes[i]
        j = i
        while j > array_offset and compare(arrays, dtypes, arrays_length, array_index, tmp, indexes[j - 1]) < 0:
            indexes[j] = indexes[j - 1]
            j -= 1
        indexes[j] = tmp
    if ranges != NULL:
        add_group(ranges, range_length, array_offset)
        for i in range(array_offset + 1, array_offset + array_length):
            cmp = compare(arrays, dtypes, arrays_length, array_index, indexes[i - 1], indexes[i])
            if cmp != 0:
                add_group(ranges, range_length, i)


cdef void radix_argsort_groups(unsigned char **arrays, int *dtypes, int arrays_length,
                               int *indexes, int array_index, int array_offset, int array_length,
                               int *ranges, int *range_length) nogil:
    radix_argsort(arrays, dtypes, arrays_length, indexes, array_index, 
                  array_offset, array_length, ranges, range_length)
    add_group(ranges, range_length, array_length)


cdef void radix_argsort(unsigned char **arrays, int *dtypes, int arrays_length,
                        int *indexes, int array_index, int array_offset, int array_length,
                        int *ranges, int *range_length) nogil:
    if array_length <= 1 or array_index >= arrays_length:
        if array_length >= 1:
            add_group(ranges, range_length, array_offset)
        return
    if array_length <= INSERTION_SORT_LIMIT:
        insertion_argsort(arrays, dtypes, arrays_length, indexes, array_index, array_offset, array_length,
                          ranges, range_length)
        return
    cdef int dtype = dtypes[array_index], string_length
    if dtype == C_ARRAY_TYPE_INT64:
        radix_argsort_int(arrays, dtypes, arrays_length, indexes, array_index, array_offset, array_length,
                          ranges, range_length)
    elif dtype == C_ARRAY_TYPE_FLOAT64:
        radix_argsort_float(arrays, dtypes, arrays_length, indexes, array_index, array_offset, array_length,
                            ranges, range_length)
    elif dtype & C_STRING_TYPE_OFFSET_MASK == C_ARRAY_TYPE_STRING:
        string_length = dtype >> C_STRING_TYPE_OFFSET_BITS
        radix_argsort_string(arrays, dtypes, arrays_length, indexes,
                             array_index, array_offset, array_length, 0, string_length * 4,
                             ranges, range_length)


cdef inline ulong convert_float(ulong a) nogil:
    if a & 0x8000000000000000l == 0:
        return a ^ 0x8000000000000000l
    return a ^ 0xffffffffffffffffl


cdef void radix_argsort_float(unsigned char **arrays, int *dtypes, int arrays_length, int *indexes,
                            int array_index, int array_offset, int array_length,
                            int *ranges, int *range_length) nogil:
    if array_length <= 1 or array_index >= arrays_length:
        if array_length >= 1:
            add_group(ranges, range_length, array_offset)
        return
    if array_length <= INSERTION_SORT_LIMIT:
        insertion_argsort(arrays, dtypes, arrays_length, indexes, array_index, 
                          array_offset, array_length, ranges, range_length)
        return
    cdef int i, index
    cdef ulong val
    cdef ulong* array = <ulong*> arrays[array_index]
    cdef ulong min_val = convert_float(array[indexes[array_offset]])
    cdef ulong max_val = convert_float(array[indexes[array_offset]])
    for i in range(array_offset + 1, array_offset + array_length):
        val = convert_float(array[indexes[i]])
        max_val = max(max_val, val)
        min_val = min(min_val, val)
    cdef ulong val_range = max_val - min_val
    cdef int value_bits = 64
    cdef int div = value_bits // 2
    while div > 0:
        if val_range >> div == 0:
            value_bits -= div
        else:
            val_range >>= div
        div >>= 1
    cdef int bits = min(value_bits, RADIX_BITS)
    cdef int shift = value_bits - bits
    cdef int bin_length = 1 << bits
    cdef int* bins = <int*> malloc((bin_length + 2) * sizeof(int))
    for i in range(bin_length + 2):
        bins[i] = 0
    cdef int bin_i
    for i in range(array_offset, array_offset + array_length):
        index = indexes[i]
        val = convert_float(array[index])
        bin_i = (val - min_val) >> shift
        bins[bin_i + 2] += 1
    for i in range(2, bin_length + 2):
        bins[i] += bins[i - 1]
    cdef int* indexes_new = <int*> malloc(array_length * sizeof(int))
    for i in range(array_offset, array_offset + array_length):
        index = indexes[i]
        val = convert_float(array[index])
        bin_i = (val - min_val) >> shift
        indexes_new[bins[bin_i + 1]] = index
        bins[bin_i + 1] += 1
    for i in range(array_length):
        indexes[i + array_offset] = indexes_new[i]
    free(indexes_new)
    if shift == 0:
        for i in range(1, bin_length + 1):
            radix_argsort(arrays, dtypes, arrays_length, indexes, array_index + 1,
                          array_offset + bins[i - 1], bins[i] - bins[i - 1],
                          ranges, range_length)
    else:
        for i in range(1, bin_length + 1):
            radix_argsort_float(arrays, dtypes, arrays_length, indexes, array_index,
                                array_offset + bins[i - 1], bins[i] - bins[i - 1],
                                ranges, range_length)
    free(bins)


cdef void radix_argsort_int(unsigned char **arrays, int *dtypes, int arrays_length, int *indexes,
                            int array_index, int array_offset, int array_length,
                            int *ranges, int *range_length) nogil:
    if array_length <= 1 or array_index >= arrays_length:
        if array_length >= 1:
            add_group(ranges, range_length, array_offset)
        return
    if array_length <= INSERTION_SORT_LIMIT:
        insertion_argsort(arrays, dtypes, arrays_length, indexes, array_index, 
                          array_offset, array_length, ranges, range_length)
        return
    cdef int i, index
    cdef long val
    cdef long* array = <long*> arrays[array_index]
    cdef long min_val = array[indexes[array_offset]], max_val = array[indexes[array_offset]]
    for i in range(array_offset + 1, array_offset + array_length):
        val = array[indexes[i]]
        if val > max_val:
            max_val = val
        if val < min_val:
            min_val = val
    cdef long val_range = max_val - min_val
    cdef int value_bits = 64
    cdef int div = value_bits // 2
    while div > 0:
        if val_range >> div == 0:
            value_bits -= div
        else:
            val_range >>= div
        div >>= 1
    cdef int bits = min(value_bits, RADIX_BITS)
    cdef int shift = value_bits - bits
    cdef int bin_length = 1 << bits
    cdef int* bins = <int*> malloc((bin_length + 2) * sizeof(int))
    for i in range(bin_length + 2):
        bins[i] = 0
    cdef int bin_i
    for i in range(array_offset, array_offset + array_length):
        index = indexes[i]
        val = array[index]
        bin_i = (val - min_val) >> shift
        bins[bin_i + 2] += 1
    for i in range(2, bin_length + 2):
        bins[i] += bins[i - 1]
    cdef int* indexes_new = <int*> malloc(array_length * sizeof(int))
    for i in range(array_offset, array_offset + array_length):
        index = indexes[i]
        val = array[index]
        bin_i = (val - min_val) >> shift
        indexes_new[bins[bin_i + 1]] = index
        bins[bin_i + 1] += 1
    for i in range(array_length):
        indexes[i + array_offset] = indexes_new[i]
    free(indexes_new)
    if shift == 0:
        for i in range(1, bin_length + 1):
            radix_argsort(arrays, dtypes, arrays_length, indexes, array_index + 1,
                          array_offset + bins[i - 1], bins[i] - bins[i - 1], 
                          ranges, range_length)
    else:
        for i in range(1, bin_length + 1):
            radix_argsort_int(arrays, dtypes, arrays_length, indexes, array_index,
                              array_offset + bins[i - 1], bins[i] - bins[i - 1],
                              ranges, range_length)
    free(bins)


cdef void radix_argsort_string(uchar **arrays, int *dtypes, int arrays_length, int *indexes,
                               int array_index, int array_offset, int array_length,
                               int string_offset, int string_length,
                               int *ranges, int *range_length) nogil:
    if array_length <= 1 or array_index >= arrays_length:
        if array_length >= 1:
            add_group(ranges, range_length, array_offset)
        return
    if string_offset >= string_length:
        radix_argsort(arrays, dtypes, arrays_length, indexes, array_index + 1, 
                      array_offset, array_length, ranges, range_length)
        return
    if array_length <= INSERTION_SORT_LIMIT:
        insertion_argsort(arrays, dtypes, arrays_length, indexes, array_index, 
                          array_offset, array_length, ranges, range_length)
        return
    cdef uchar* array = <uchar*> arrays[array_index]
    cdef uchar min_val = array[indexes[array_offset] * string_length + string_offset]
    cdef uchar max_val = array[indexes[array_offset] * string_length + string_offset]
    cdef uchar val
    cdef int i
    for i in range(array_offset + 1, array_offset + array_length):
        val = array[indexes[i] * string_length + string_offset]
        max_val = max(max_val, val)
        min_val = min(min_val, val)
    cdef ulong val_range = max_val - min_val
    if val_range == 0 and max_val == 0:
        radix_argsort(arrays, dtypes, arrays_length, indexes, array_index + 1, 
                      array_offset, array_length, ranges, range_length)
        return
    cdef int value_bits = 8,
    cdef int div = value_bits // 2
    while div > 0:
        if val_range >> div == 0:
            value_bits -= div
        else:
            val_range >>= div
        div >>= 1
    cdef int bits = value_bits, index
    cdef int bin_length = 1 << bits
    cdef int* bins = <int*> malloc((bin_length + 2) * sizeof(int))
    for i in range(bin_length + 2):
        bins[i] = 0
    cdef uchar bin_i
    for i in range(array_offset, array_offset + array_length):
        index = indexes[i]
        bin_i = array[index * string_length + string_offset] - min_val
        bins[bin_i + 2] += 1
    for i in range(2, bin_length + 2):
        bins[i] += bins[i - 1]
    cdef int* indexes_new = <int*> malloc(array_length * sizeof(int))
    for i in range(array_offset, array_offset + array_length):
        index = indexes[i]
        bin_i = array[index * string_length + string_offset] - min_val
        indexes_new[bins[bin_i + 1]] = index
        bins[bin_i + 1] += 1
    for i in range(array_length):
        indexes[i + array_offset] = indexes_new[i]
    free(indexes_new)
    cdef int start = 1
    if min_val == 0:
        radix_argsort(arrays, dtypes, arrays_length, indexes, array_index + 1, 
                      array_offset + bins[0], bins[1] - bins[0], ranges, range_length)
        start = 2
    for i in range(start, bin_length + 1):
        radix_argsort_string(arrays, dtypes, arrays_length, indexes, array_index,
                             array_offset + bins[i - 1], bins[i] - bins[i - 1], 
                             string_offset + 4, string_length, ranges, range_length)
    free(bins)


cdef int* unwrap_arrays(arrays: List[np.ndarray], unsigned char **c_arrays):
    data, dtypes = convert_to_uint8(arrays)
    cdef int[::1] dtypes_mem = np.array(dtypes, dtype=np.int32)
    cdef unsigned char[::1] mem
    for i in range(len(arrays)):
        mem = data[i]
        c_arrays[i] = &mem[0]
    return &dtypes_mem[0]


def radix_argsort_py(arrays: List[np.ndarray]):
    cdef unsigned char **c_arrays = <unsigned char **> malloc(len(arrays) * sizeof(unsigned char *))
    cdef int *dtypes_mem = unwrap_arrays(arrays, c_arrays)
    cdef int[::1] indexes = np.arange(len(arrays[0]), dtype=np.int32)
    radix_argsort(c_arrays, dtypes_mem, len(arrays), &indexes[0], 0, 0, len(arrays[0]), NULL, NULL)
    free(c_arrays)
    return indexes


def compare_py(arrays: List[np.ndarray], l: int, r: int):
    cdef unsigned char **c_arrays = <unsigned char **> malloc(len(arrays) * sizeof(unsigned char *))
    cdef int *dtypes_mem = unwrap_arrays(arrays, c_arrays)
    res = compare(c_arrays, dtypes_mem, len(arrays), 0, l, r)
    free(c_arrays)
    return res

def insertion_argsort_py(arrays: List[np.ndarray]):
    cdef unsigned char **c_arrays = <unsigned char **> malloc(len(arrays) * sizeof(unsigned char *))
    cdef int *dtypes_mem = unwrap_arrays(arrays, c_arrays)
    cdef int[::1] indexes = np.arange(len(arrays[0]), dtype=np.int32)
    insertion_argsort(c_arrays, dtypes_mem, len(arrays), &indexes[0], 0, 0, len(arrays[0]), NULL, NULL)
    free(c_arrays)
    return indexes

