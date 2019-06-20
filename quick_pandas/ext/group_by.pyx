# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True


from typing import List, Callable

import pandas as pd

from quick_pandas.ext.argsort cimport radix_argsort, radix_argsort_groups, unwrap_arrays, compare, C_ARRAY_TYPE_INT64, C_ARRAY_TYPE_FLOAT64
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from quick_pandas.np_funcs import *
from cython.parallel import prange

ctypedef unsigned char uchar
ctypedef fused numeric:
    double
    long
cdef int C_NP_FUNCS_SUM = NP_FUNCS_SUM
cdef int C_NP_FUNCS_MEAN = NP_FUNCS_MEAN

cdef inline numeric mean0(numeric *values_c, int *indexes, int start, int end) nogil:
    cdef numeric res = 0
    cdef int i
    for i in range(start, end):
        res += values_c[indexes[i]]
    return <numeric>(res / (end - start))


cdef inline numeric sum0(numeric *values_c, int *indexes, int start, int end) nogil:
    cdef numeric res = 0
    cdef int i
    for i in range(start, end):
        res += values_c[indexes[i]]
    return res


cdef void transform(uchar **values_c, int *values_types, int* range_start, int range_size,
                    uchar **new_values_c, int *new_values_types,
                    int func, int arrays_length, int *indexes) nogil:
    cdef int i, j, k, start, end, dtype
    cdef long sum_long, mean_long
    cdef double sum_double, mean_double
    for k in range(arrays_length):
        for i in prange(range_size - 1, nogil=True):
            dtype = values_types[k]
            start = range_start[i]
            end = range_start[i + 1]
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


cdef void get_groups(uchar **arrays, int *dtypes, int arrays_length, int array_length,
                     int *indexes, int *range_start, int *range_length) nogil:
    cdef int i
    range_start[0] = 0
    range_length[0] = 1
    for i in range(1, array_length):
        if compare(arrays, dtypes, arrays_length, 0, indexes[i], indexes[i - 1]) != 0:
            range_start[range_length[0]] = i
            range_length[0] += 1
    range_start[range_length[0]] = array_length
    range_length[0] += 1

def group_and_transform(df: pd.DataFrame, by_columns: List[str], targets: List[str], func: Callable,
                        sort: bool = False, inplace=False):
    keys = [df[c].values for c in by_columns]
    array_length = len(keys[0])
    arrays_length = len(keys)
    cdef unsigned char **c_arrays = <unsigned char **> malloc(len(keys) * sizeof(unsigned char *))
    cdef int[::1] indexes = np.arange(len(keys[0]), dtype=np.int32)
    cdef int *dtypes_mem = unwrap_arrays(keys, c_arrays)
    cdef int i, range_size = 0, start, end
    cdef int* range_start = <int *> malloc((array_length + 1) * sizeof(int))
    printf('[debug] radix sort starts\n')
    radix_argsort_groups(c_arrays, dtypes_mem, arrays_length, &indexes[0], 0, 0, array_length, 
                         range_start, &range_size)
    printf('[debug] radix sort completes, group size: %d\n', range_size)

    func = NP_FUNCS_MAP_REVERSE.get(func, None)
    if func is None:
        raise Exception('unsupported transform function: %s' % func)
    values = [df[c].values for c in targets]
    new_values = [np.empty_like(v) for v in values]
    cdef unsigned char **values_c = <unsigned char **> malloc(len(values) * sizeof(unsigned char *))
    cdef int* values_types = unwrap_arrays(values, values_c)
    cdef unsigned char **new_values_c = <unsigned char **> malloc(len(values) * sizeof(unsigned char *))
    cdef int* new_values_types = unwrap_arrays(new_values, new_values_c)
    printf('[debug] prepare for transform completes\n')
    transform(values_c, values_types, range_start, range_size,
              new_values_c, new_values_types, func, arrays_length, &indexes[0])
    printf('[debug] transform completes\n')
    free(new_values_c)
    free(values_c)
    free(range_start)
    free(c_arrays)
    #data = {**dict(zip(by_columns, keys)), **dict(zip(targets, new_values))}
    #return pd.DataFrame(data)
    for i in range(len(targets)):
        df['%s_t' % targets[i]] = new_values[i]
    printf('[debug] dataframe completes\n')
    return df


def transform_py(values: List[np.ndarray], ranges, indexes: np.ndarray, func: Callable):
    cdef uchar **c_arrays = <uchar **> malloc(len(values) * sizeof(uchar *))
    cdef int* dtypes_mem = unwrap_arrays(values, c_arrays)
    cdef int* range_start = <int *> malloc(len(indexes) * sizeof(int *))
    cdef int range_size = 1, arrays_length = len(values), array_length = len(values[0])
    range_start[0] = 0
    for r in ranges:
        range_start[range_size] = r[0]
        range_size += 1
    range_start[range_size] = array_length
    range_size += 1
    cdef uchar **new_values_c = <uchar **> malloc(len(values) * sizeof(uchar *))
    new_values = [np.empty_like(v) for v in values]
    cdef int* new_values_types = unwrap_arrays(new_values, new_values_c)
    cdef int func_int = NP_FUNCS_MAP_REVERSE.get(func, None)
    cdef int [::1] ii = indexes
    transform(c_arrays, dtypes_mem, range_start, range_size,
              new_values_c, new_values_types, func_int, len(values), &ii[0])
    free(new_values_c)
    free(range_start)
    free(c_arrays)
    return new_values

