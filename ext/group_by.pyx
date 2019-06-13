# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True


from libc.stdio cimport printf
from typing import List, Callable

import pandas as pd

from ext.argsort cimport radix_argsort, unwrap_arrays, compare, C_ARRAY_TYPE_INT64, C_ARRAY_TYPE_FLOAT64
from libc.stdlib cimport malloc, free
from quick_pandas.np_funcs import *

ctypedef unsigned char uchar
ctypedef fused numeric:
    double
    long
cdef int C_NP_FUNCS_SUM = NP_FUNCS_SUM
cdef int C_NP_FUNCS_MEAN = NP_FUNCS_MEAN

cdef numeric mean0(numeric *values_c, int[::1] indexes, int start, int end) nogil:
    cdef numeric res = 0
    cdef int i
    for i in range(start, end):
        res += values_c[indexes[i]]
    return <numeric>(res / (end - start))


cdef numeric sum0(numeric *values_c, int[::1] indexes, int start, int end) nogil:
    cdef numeric res = 0
    cdef int i
    for i in range(start, end):
        res += values_c[indexes[i]]
    return res



cdef void transform(uchar **values_c, int[::1] values_types, int* range_start, int* range_end, int range_size,
                    uchar **new_values_c, int[::1] new_values_types,
                    int func, int arrays_length, int[::1] indexes) nogil:
    cdef int i, j, start, end, dtype
    cdef long sum_long, mean_long
    cdef double sum_double, mean_double
    for k in range(arrays_length):
        for i in range(range_size):
            dtype = values_types[k]
            start = range_start[i]
            end = range_end[i]
            if func == C_NP_FUNCS_SUM:
                if dtype == C_ARRAY_TYPE_INT64:
                    sum_long = sum0(<long*>values_c[k], indexes, start, end)
                    for j in range(start, end):
                        (<long*>new_values_c[k])[indexes[j]] = sum_long
                elif dtype == C_ARRAY_TYPE_FLOAT64:
                    sum_double = sum0(<double*>values_c[k], indexes, start, end)
                    for j in range(start, end):
                        (<double*>new_values_c[k])[indexes[j]] = sum_double
            elif func == C_NP_FUNCS_MEAN:
                if dtype == C_ARRAY_TYPE_INT64:
                    mean_long = mean0(<long*>values_c[k], indexes, start, end)
                    for j in range(start, end):
                        (<long*>new_values_c[k])[indexes[j]] = mean_long
                elif dtype == C_ARRAY_TYPE_FLOAT64:
                    mean_double = mean0(<double*>values_c[k], indexes, start, end)
                    for j in range(start, end):
                        (<double*>new_values_c[k])[indexes[j]] = mean_double


def group_and_transform(df: pd.DataFrame, by_columns: List[str], targets: List[str], func: Callable,
                        sort: bool = False, inplace=False):
    keys = [df[c].values for c in by_columns]
    array_length = len(keys[0])
    arrays_length = len(keys)
    cdef unsigned char **c_arrays = <unsigned char **> malloc(len(keys) * sizeof(unsigned char *))
    indexes = np.arange(len(keys[0]), dtype=np.int32)
    cdef int[::1] dtypes_mem = unwrap_arrays(keys, c_arrays)
    radix_argsort(c_arrays, dtypes_mem, len(keys), indexes, 0, 0, len(keys[0]))
    cdef int i, range_size = 0, start, end
    cdef int* range_start = <int *> malloc(array_length * sizeof(int *))
    cdef int* range_end = <int *> malloc(array_length * sizeof(int *))
    range_start[0] = 0
    for i in range(1, array_length):
        if compare(c_arrays, dtypes_mem, arrays_length, 0, indexes[i], indexes[i - 1]) != 0:
            range_end[range_size] = i
            range_start[range_size + 1] = i
            range_size += 1
    range_end[range_size] = array_length
    range_size += 1

    func = NP_FUNCS_MAP_REVERSE.get(func, None)
    if func is None:
        raise Exception('unsupported transform function: %s' % func)
    values = [df[c].values for c in targets]
    new_values = [np.empty_like(v) for v in values]
    cdef unsigned char **values_c = <unsigned char **> malloc(len(values) * sizeof(unsigned char *))
    cdef int[::1] values_types = unwrap_arrays(values, values_c)
    cdef unsigned char **new_values_c = <unsigned char **> malloc(len(values) * sizeof(unsigned char *))
    cdef int[::1] new_values_types = unwrap_arrays(new_values, new_values_c)
    transform(values_c, values_types, range_start, range_end, range_size,
              new_values_c, new_values_types, func, arrays_length, indexes)

    data = {**dict(zip(by_columns, keys)), **dict(zip(targets, new_values))}
    free(new_values_c)
    free(values_c)
    free(range_start)
    free(range_end)
    free(c_arrays)
    return pd.DataFrame(data)


def transform_py(values: List[np.ndarray], ranges, indexes: np.ndarray, func: Callable):
    cdef uchar **c_arrays = <uchar **> malloc(len(values) * sizeof(uchar *))
    cdef int[::1] dtypes_mem = unwrap_arrays(values, c_arrays)
    cdef int* range_start = <int *> malloc(len(indexes) * sizeof(int *))
    cdef int* range_end = <int *> malloc(len(indexes) * sizeof(int *))
    cdef int range_size = 0, arrays_length = len(values), array_length = len(values[0])
    for r in ranges:
        range_start[range_size] = r[0]
        range_end[range_size] = r[1]
        range_size += 1
    cdef uchar **new_values_c = <uchar **> malloc(len(values) * sizeof(uchar *))
    new_values = [np.empty_like(v) for v in values]
    cdef int[::1] new_values_types = unwrap_arrays(new_values, new_values_c)
    cdef int func_int = NP_FUNCS_MAP_REVERSE.get(func, None)
    transform(c_arrays, dtypes_mem, range_start, range_end, range_size,
              new_values_c, new_values_types, func_int, len(values), indexes)
    free(new_values_c)
    free(range_start)
    free(range_end)
    free(c_arrays)
    return new_values
