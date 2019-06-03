from typing import List

import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import Dict

from quick_pandas import dtypes
from quick_pandas.sort import radix_argsort0_int


@njit()
def update_value():
    pass


@njit()
def group_and_transform0(keys: List[np.ndarray], vals: List[np.ndarray]):
    key_length = len(keys)
    val_length = len(vals)
    length = len(keys[0])
    indexes = np.arange(length)
    radix_argsort0_int(keys[0], indexes, 0, length)
    new_vals = [np.empty_like(vals[i]) for i in range(val_length)]
    for i in range(val_length):
        new_vals[i][0] = vals[i][indexes[0]]
    pre_index = 0
    for i in range(1, length):
        for k in range(val_length):
            new_vals[k][i] = vals[k][indexes[i]]
        same = True
        for j in range(key_length):
            key = keys[0]
            if key[indexes[pre_index]] != key[indexes[i]]:
                same = False
                break
        if not same:
            for k in range(val_length):
                res = np.mean(new_vals[k][pre_index: i])
                for j in range(pre_index, i):
                    vals[k][indexes[j]] = res
                    new_vals[k][j] = res
            pre_index = i
    for k in range(val_length):
        res = np.mean(new_vals[k][pre_index: length])
        for j in range(pre_index, length):
            vals[k][indexes[j]] = res
            new_vals[k][j] = res

    return new_vals


int_array = types.int64[:]


@njit()
def group(keys_list: List[np.ndarray], keys_dtype: List[int], keys_index: int, indexes: List[int]):
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


def group_and_transform(df: pd.DataFrame, by: List[str], sort: bool = False):
    targets = [name for name in df.columns if name not in by]
    keys = [df[c].values for c in by]
    values = [df[c].values for c in targets]
    sroted = group_and_transform0(keys, values)
    data = {**dict(zip(by, keys)),
            **dict(zip(targets, values))}
    res = pd.DataFrame(data)
    print(res)

    res = pd.DataFrame(data)
