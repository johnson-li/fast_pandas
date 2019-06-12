from typing import List, Callable

import pandas as pd
from numba import njit, types
from numba.typed import Dict

from quick_pandas import dtypes
from quick_pandas.dtypes import convert_to_uint8
from quick_pandas.np_funcs import *
from quick_pandas.sort import radix_argsort0_mix


@njit()
def group_and_transform0(keys: List[np.ndarray], dts: List[int], vals: List[np.ndarray], length: int, func: int,
                         sort=False, inplace=False):
    val_length = len(vals)
    indexes = np.arange(length)
    groups = radix_argsort0_mix(keys, dts, indexes)
    # new_vals = [np.empty_like(vals[i]) for i in range(val_length)]
    res_vals = [np.empty_like(vals[i]) for i in range(val_length)]
    # for i in range(val_length):
    #     new_vals[i][0] = vals[i][indexes[0]]
    # for start, end, _, _ in groups:
    #     for k in range(val_length):
    #         for j in range(start, end):
    #             new_vals[k][j] = vals[k][indexes[j]]
    #         res = new_vals[k][0]
    #         if func == NP_FUNCS_MEAN:
    #             res = np.mean(new_vals[k][start: end])
    #         elif func == NP_FUNCS_SUM:
    #             res = np.sum(new_vals[k][start: end])
    #         elif func == NP_FUNCS_MEDIAN:
    #             res = np.median(new_vals[k][start: end])
    #         for j in range(start, end):
    #             res_vals[k][indexes[j]] = res
    #             if inplace:
    #                 vals[k][indexes[j]] = res
    return res_vals


int_array = types.int64[:]


@njit()
def group(keys_list: List[np.ndarray], keys_dtype: List[int], keys_index: int, indexes: List[int], inplace=False):
    if keys_index >= len(keys_list):
        return [indexes]
    keys = keys_list[keys_index]
    dtype = keys_dtype[keys_index]
    if dtype == dtypes.ARRAY_TYPE_INT64:
        keys_int = keys.view(np.int64)
        groups_int = Dict.empty(key_type=types.int64, value_type=int_array)
        for index in indexes:
            key = keys_int[index]
            key = types.int64(0)
            groups_int.setdefault(key, [])
            print(groups_int[0])
            groups_int[0] += [index]
        res = []
        for val in groups_int.values():
            res.extend(group(keys_list, keys_dtype, keys_index + 1, val))
        return res
    elif dtype == dtypes.ARRAY_TYPE_FLOAT64:
        keys_float = keys.view(np.float64)
        groups_float = Dict.empty(key_type=types.float64, value_type=int_array)
        for index in indexes:
            key = keys_float[index]
            groups_float.setdefault(key, [])
            groups_float[key].append(index)
        res = []
        for val in groups_float.values():
            res.extend(group(keys_list, keys_dtype, keys_index + 1, val))
        return res
    elif dtype & dtypes.STRING_TYPE_OFFSET_MASK == dtypes.ARRAY_TYPE_STRING:
        pass


def group_and_transform(df: pd.DataFrame, by: List[str], targets: List[str], func: Callable,
                        sort: bool = False, inplace=False):
    keys = [df[c].values for c in by]
    values = [df[c].values for c in targets]
    au8, dts = convert_to_uint8(keys)
    func = NP_FUNCS_MAP_REVERSE.get(func, None)
    if func is None:
        raise Exception('unsupported transform function: %s' % func)
    new_vals = group_and_transform0(au8, dts, values, len(keys[0]), func, sort=sort, inplace=inplace)
    data = {**dict(zip(by, keys)),
            **dict(zip(targets, new_vals))}
    return pd.DataFrame(data)
