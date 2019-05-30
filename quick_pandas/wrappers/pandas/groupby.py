from typing import List

import numba
import numpy as np
import pandas as pd
from numba import njit

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
