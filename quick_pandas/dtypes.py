from typing import List

import numpy as np

ARRAY_TYPE_STRING = 0
ARRAY_TYPE_FLOAT32 = 1
ARRAY_TYPE_FLOAT64 = 2
ARRAY_TYPE_INT32 = 3
ARRAY_TYPE_INT64 = 4

STRING_TYPE_OFFSET_BITS = 8
STRING_TYPE_OFFSET_MASK = 0xff


def convert_to_uint8(arrays: List[np.ndarray]):
    return [(a if not a.dtype == object else a.astype(str)).view(np.uint8) for a in arrays], get_dtypes(arrays)


def get_dtypes(arrays: List[np.ndarray]):
    return [dtype_numeric(a) for a in arrays]


def dtype_numeric(array: np.ndarray):
    if array.dtype == object:
        array = array.astype(str)
    dtype = array.dtype
    if dtype.type == np.str_:
        res = ARRAY_TYPE_STRING
        res |= (array.itemsize // 4) << STRING_TYPE_OFFSET_BITS
        return res
    if dtype.type == np.float64:
        return ARRAY_TYPE_FLOAT64
    if dtype.type == np.float32:
        return ARRAY_TYPE_FLOAT32
    if dtype.type == np.int32:
        return ARRAY_TYPE_INT32
    if dtype.type == np.int64:
        return ARRAY_TYPE_INT64
    raise Exception('unknown dtype: %s' % dtype)
