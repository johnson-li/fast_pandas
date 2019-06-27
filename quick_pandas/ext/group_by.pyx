# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True


from typing import List, Callable

import pandas as pd

from quick_pandas.ext.argsort cimport radix_argsort, radix_argsort_groups, unwrap_arrays, compare, C_ARRAY_TYPE_INT64, C_ARRAY_TYPE_FLOAT64, C_ARRAY_TYPE_INT32, C_ARRAY_TYPE_FLOAT32

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from quick_pandas.np_funcs import *
from cython.parallel import prange
from quick_pandas.ext.operators cimport OPERATOR_INT, OPERATOR_LONG, OPERATOR_FLOAT, OPERATOR_DOUBLE, sum_int, sum_long, sum_float, sum_double, mean_int, mean_long, mean_float, mean_double

ctypedef unsigned char uchar
cdef int C_NP_FUNCS_SUM = NP_FUNCS_SUM
cdef int C_NP_FUNCS_MEAN = NP_FUNCS_MEAN


cdef void transform(uchar **values_c, int *values_types, int* range_start, int range_size,
                    uchar **new_values_c, int *new_values_types,
                    int func, int arrays_length, int *indexes, 
                    OPERATOR_INT op_int, OPERATOR_LONG op_long, OPERATOR_FLOAT op_float, OPERATOR_DOUBLE op_double) nogil:
    cdef int i, j, k, start, end, dtype
    cdef int res_int, *new_values_int, *values_int
    cdef long res_long, *new_values_long, *values_long
    cdef float res_float, *new_values_float, *values_float
    cdef double res_double, *new_values_double, *values_double
    for k in range(arrays_length):
        dtype = values_types[k]
        if dtype == C_ARRAY_TYPE_INT64:
            values_long = <long*>values_c[k]
            new_values_long = <long*>new_values_c[k]
            for i in range(range_size - 1):
                start = range_start[i]
                end = range_start[i + 1]
                res_long = op_long(values_long, indexes, start, end)
                for j in range(start, end):
                    new_values_long[indexes[j]] = res_long
        elif dtype == C_ARRAY_TYPE_INT32:
            values_int = <int*>values_c[k]
            new_values_int = <int*>new_values_c[k]
            for i in range(range_size - 1):
                start = range_start[i]
                end = range_start[i + 1]
                res_int = op_int(values_int, indexes, start, end)
                for j in range(start, end):
                    new_values_int[indexes[j]] = res_int
        elif dtype == C_ARRAY_TYPE_FLOAT64:
            values_double = <double*>values_c[k]
            new_values_double = <double*>new_values_c[k]
            for i in range(range_size - 1):
                start = range_start[i]
                end = range_start[i + 1]
                res_double = op_double(values_double, indexes, start, end)
                for j in range(start, end):
                    new_values_double[indexes[j]] = res_double
        elif dtype == C_ARRAY_TYPE_FLOAT32:
            values_float = <float*>values_c[k]
            new_values_float = <float*>new_values_c[k]
            for i in range(range_size - 1):
                start = range_start[i]
                end = range_start[i + 1]
                res_float = op_float(values_float, indexes, start, end)
                for j in range(start, end):
                    new_values_float[indexes[j]] = res_float


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


cdef void get_ops(int func, OPERATOR_INT *o1, OPERATOR_LONG *o2, OPERATOR_FLOAT *o3, OPERATOR_DOUBLE *o4) nogil:
    if func == 0:
        o1[0] = mean_int
        o2[0] = mean_long
        o3[0] = mean_float
        o4[0] = mean_double
    elif func == 1:
        o1[0] = sum_int
        o2[0] = sum_long
        o3[0] = sum_float
        o4[0] = sum_double


def group_and_transform(df: pd.DataFrame, by_columns: List[str], targets: List[str], func: Callable,
                        sort=False, inplace=False):
    return group_and_transform0(df, by_columns, targets, func, NULL, NULL, NULL, NULL, sort, inplace)


cdef group_and_transform0(df: pd.DataFrame, by_columns: List[str], targets: List[str], func: Callable,
                          OPERATOR_INT op_int, OPERATOR_LONG op_long,
                          OPERATOR_FLOAT op_float, OPERATOR_DOUBLE op_double, sort: bool, inplace: bool):
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
    if op_int == NULL or op_long == NULL:
        func = NP_FUNCS_MAP_REVERSE.get(func, None)
        get_ops(func, &op_int, &op_long, &op_float, &op_double)
    if op_int == NULL or op_long == NULL:
        raise Exception('unsupported transform function: %s' % func)
    values = [df[c].values for c in targets]
    new_values = [np.empty_like(v) for v in values]
    cdef unsigned char **values_c = <unsigned char **> malloc(len(values) * sizeof(unsigned char *))
    cdef int* values_types = unwrap_arrays(values, values_c)
    cdef unsigned char **new_values_c = <unsigned char **> malloc(len(values) * sizeof(unsigned char *))
    cdef int* new_values_types = unwrap_arrays(new_values, new_values_c)
    printf('[debug] prepare for transform completes\n')
    transform(values_c, values_types, range_start, range_size,
              new_values_c, new_values_types, func, arrays_length, &indexes[0], op_int, op_long, op_float, op_double)
    printf('[debug] transform completes\n')
    # for i in range(len(targets)):
    #     df['%s_t' % targets[i]] = new_values[i]
    printf('[debug] dataframe completes\n')
    free(new_values_c)
    free(values_c)
    free(range_start)
    free(c_arrays)
    return new_values
    #data = {**dict(zip(by_columns, keys)), **dict(zip(targets, new_values))}
    #return pd.DataFrame(data)


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
    cdef OPERATOR_INT op_int = NULL
    cdef OPERATOR_LONG op_long = NULL
    cdef OPERATOR_FLOAT op_float = NULL
    cdef OPERATOR_DOUBLE op_double = NULL
    get_ops(func_int, &op_int, &op_long, &op_float, &op_double)
    transform(c_arrays, dtypes_mem, range_start, range_size,
              new_values_c, new_values_types, func_int, len(values), &ii[0], op_int, op_long, op_float, op_double)
    free(new_values_c)
    free(range_start)
    free(c_arrays)
    return new_values

