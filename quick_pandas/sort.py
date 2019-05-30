from typing import List

import numpy as np
from numba import njit

from quick_pandas import dtypes

INSERTION_SORT_LIMIT = 64
RADIX_BITS = 8
NUMBA_CACHE = False


@njit(cache=NUMBA_CACHE)
def radix_sort0(array: np.ndarray, array_offset: int, array_length: int):
    if array_length == 0:
        return
    if array_length <= INSERTION_SORT_LIMIT:
        for i in range(array_offset + 1, array_offset + array_length):
            val = array[i]
            j = i
            while j > array_offset and val < array[j - 1]:
                array[j] = array[j - 1]
                j -= 1
            array[j] = val
        return

    min_val = np.min(array[array_offset: array_offset + array_length])
    max_val = np.max(array[array_offset: array_offset + array_length])
    length = max_val - min_val
    value_bits = 64
    div = value_bits // 2
    while div > 0:
        if length >> div == 0:
            value_bits -= div
        else:
            length >>= div
        div >>= 1
    bits = min(value_bits, RADIX_BITS)
    last = bits == value_bits
    shift = value_bits - bits
    bin_length = 1 << bits
    bins = np.zeros(bin_length + 1, np.int32)
    for val in array[array_offset: array_offset + array_length]:
        bin_i = (val - min_val) >> shift
        bins[bin_i + 1] += 1
    for i in range(1, bin_length + 1):
        bins[i] += bins[i - 1]
    count = bins.copy()
    new_array = np.zeros_like(array[array_offset: array_offset + array_length])
    for val in array[array_offset: array_offset + array_length]:
        bin_i = (val - min_val) >> shift
        index = count[bin_i]
        count[bin_i] += 1
        new_array[index] = val
    array[array_offset:array_offset + array_length] = new_array
    if not last:
        for i in range(1, bin_length + 1):
            radix_sort0(array, array_offset + bins[i - 1], bins[i] - bins[i - 1])


def radix_sort(array: np.ndarray):
    radix_sort0(array, 0, len(array))


@njit(cache=NUMBA_CACHE)
def radix_argsort0_int(array_list: List[np.ndarray], array_type_list: List[int], array_index: int,
                       array: np.ndarray, indexes: np.ndarray, array_offset: int, array_length: int):
    if array_length == 0:
        return
    if array_length <= INSERTION_SORT_LIMIT:
        return insertion_argsort0(array_list, array_type_list, array_index, indexes, array_offset, array_length)
    min_val = array[indexes[array_offset]]
    max_val = array[indexes[array_offset]]
    for index in indexes[array_offset + 1: array_offset + array_length]:
        if array[index] > max_val:
            max_val = array[index]
        if array[index] < min_val:
            min_val = array[index]
    length = max_val - min_val
    value_bits = array.itemsize * 8
    div = value_bits // 2
    while div > 0:
        if length >> div == 0:
            value_bits -= div
        else:
            length >>= div
        div >>= 1
    bits = min(value_bits, RADIX_BITS)
    last = bits == value_bits
    shift = value_bits - bits
    bin_length = 1 << bits
    bins = np.zeros(bin_length + 1, np.int32)
    for index in indexes[array_offset: array_offset + array_length]:
        val = array[index]
        bin_i = (val - min_val) >> shift
        bins[bin_i + 1] += 1
    for i in range(1, bin_length + 1):
        bins[i] += bins[i - 1]
    count = bins.copy()
    new_indexes = np.zeros_like(indexes[array_offset: array_offset + array_length])
    for val in indexes[array_offset: array_offset + array_length]:
        bin_i = (array[val] - min_val) >> shift
        index = count[bin_i]
        count[bin_i] += 1
        new_indexes[index] = val
    indexes[array_offset:array_offset + array_length] = new_indexes
    if last:
        for i in range(1, bin_length + 1):
            radix_argsort0_mix(array_list, array_type_list, array_index + 1, indexes,
                               array_offset + bins[i - 1], bins[i] - bins[i - 1])
    else:
        for i in range(1, bin_length + 1):
            radix_argsort0_int(array_list, array_type_list, array_index, array, indexes,
                               array_offset + bins[i - 1], bins[i] - bins[i - 1])


@njit(cache=NUMBA_CACHE)
def array_cmp_lt(a, b, offset, length, unicode):
    if length == 0:
        return False
    if a[offset] < b[offset]:
        return True
    elif a[offset] > b[offset]:
        return False
    elif (not unicode or (offset % 4) == 0) and a[offset] == 0:
        if b[offset] != 0:
            return True
        else:
            return False
    return array_cmp_lt(a, b, offset + 1, length - 1, unicode)


@njit(cache=NUMBA_CACHE)
def radix_argsort0_str(array: np.ndarray, indexes: np.ndarray, array_offset: int, array_length: int, str_offset: int,
                       unicode: bool):
    if array_length == 0:
        return
    if str_offset >= len(array[0]):
        return
    if array_length < INSERTION_SORT_LIMIT:
        for i in range(array_offset + 1, array_offset + array_length):
            val = indexes[i]
            j = i
            while j > array_offset and array_cmp_lt(array[val], array[indexes[j - 1]], str_offset,
                                                    array.shape[1] - str_offset, unicode):
                indexes[j] = indexes[j - 1]
                j -= 1
            indexes[j] = val
        return

    min_val = array[indexes[array_offset]][str_offset]
    max_val = array[indexes[array_offset]][str_offset]
    for index in indexes[array_offset: array_offset + array_length]:
        val = array[index][str_offset]
        if val < min_val:
            min_val = val
        elif val > max_val:
            max_val = val
    if min_val == max_val:
        if max_val == 0 and (unicode or str_offset % 4) == 0:
            return
        else:
            return radix_argsort0_str(array, indexes, array_offset, array_length, str_offset + 1, unicode)
    length = max_val - min_val
    value_bits = 8
    div = value_bits // 2
    while div > 0:
        if length >> div == 0:
            value_bits -= div
        else:
            length >>= div
        div >>= 1
    bins = np.zeros((1 << value_bits) + 1, np.int32)
    for index in indexes[array_offset: array_offset + array_length]:
        val = array[index][str_offset]
        bin_i = val - min_val
        bins[bin_i + 1] += 1
    for i in range(1, len(bins)):
        bins[i] += bins[i - 1]
    count = bins.copy()
    new_indexes = np.zeros_like(indexes[array_offset: array_offset + array_length])
    for val in indexes[array_offset: array_offset + array_length]:
        bin_i = array[val][str_offset] - min_val
        index = count[bin_i]
        count[bin_i] += 1
        new_indexes[index] = val
    indexes[array_offset:array_offset + array_length] = new_indexes
    for i in range(1, len(bins)):
        if i == 1 and min_val == 0 and (unicode or str_offset % 4 == 0):
            continue
        radix_argsort0_str(array, indexes, array_offset + bins[i - 1], bins[i] - bins[i - 1], str_offset + 1, unicode)


@njit(cache=NUMBA_CACHE)
def convert_float64(item):
    if (item & np.uint64(0x8000000000000000)) == 0:
        return item ^ np.uint64(0x8000000000000000)
    return item ^ np.uint64(0xffffffffffffffff)


@njit(cache=NUMBA_CACHE)
def convert_float32(item):
    if (item & np.uint32(0x80000000)) == 0:
        return item ^ np.uint32(0x80000000)
    return item ^ np.uint32(0xffffffff)


@njit(cache=NUMBA_CACHE)
def radix_argsort0_float(array_float: np.ndarray, array: np.ndarray, indexes: np.ndarray, array_offset: int,
                         array_length: int, value_bits: int):
    if array_length <= 1 or value_bits == 0:
        return
    double = array.itemsize == 8
    if array_length <= INSERTION_SORT_LIMIT:
        for i in range(array_offset + 1, array_offset + array_length):
            index = indexes[i]
            j = i
            now = array_float[index]
            pre = array_float[indexes[j - 1]]
            while j > array_offset and (now < pre or (np.isnan(pre) and not np.isnan(now))):
                indexes[j] = indexes[j - 1]
                j -= 1
                pre = array_float[indexes[j - 1]]
            indexes[j] = index
        return
    value_bits_total = array.itemsize * 8
    value_mask = np.uint64(-1) >> np.uint64(value_bits_total - value_bits) \
        if double else np.uint32(-1) >> np.uint32(value_bits_total - value_bits)
    bits = min(value_bits, RADIX_BITS)
    shift = value_bits - bits
    bin_length = 1 << bits
    bins = np.zeros(bin_length + 1, np.int32)
    for index in indexes[array_offset: array_offset + array_length]:
        array_value = array[index]
        if double:
            array_value = convert_float64(array_value) & value_mask
        else:
            array_value = convert_float32(array_value) & value_mask
        bin_i = array_value >> np.uint32(shift)
        bins[bin_i + 1] += 1
    for i in range(1, bin_length + 1):
        bins[i] += bins[i - 1]
    count = bins.copy()
    new_indexes = np.zeros_like(indexes[array_offset: array_offset + array_length])
    for val in indexes[array_offset: array_offset + array_length]:
        array_value = array[val]
        if double:
            array_value = convert_float64(array_value) & value_mask
        else:
            array_value = convert_float32(array_value) & value_mask
        bin_i = array_value >> np.uint32(shift)
        index = count[bin_i]
        count[bin_i] += 1
        new_indexes[index] = val
    indexes[array_offset:array_offset + array_length] = new_indexes
    for i in range(1, bin_length + 1):
        radix_argsort0_float(array_float, array, indexes, array_offset + bins[i - 1], bins[i] - bins[i - 1], shift)


@njit(cache=NUMBA_CACHE)
def cmp_mix(array_list: List[np.ndarray], dts: List[int], array_index: int, l: int, r: int, str_offset=0):
    if array_index >= len(array_list):
        return 0
    array = array_list[array_index]
    dtype = dts[array_index]
    if dtype == dtypes.ARRAY_TYPE_INT64:
        array_int64 = array.view(np.int64)
        val_l = array_int64[l]
        val_r = array_int64[r]
        if val_l < val_r:
            return -1
        if val_l > val_r:
            return 1
        return cmp_mix(array_list, dts, array_index + 1, l, r)
    if dtype == dtypes.ARRAY_TYPE_FLOAT64:
        array_float64 = array.view(np.float64)
        val_l = array_float64[l]
        val_r = array_float64[r]
        nan_l = np.isnan(val_l)
        nan_r = np.isnan(val_r)
        if nan_l and not nan_r or val_l > val_r:
            return 1
        if nan_r and not nan_l or val_l < val_r:
            return -1
        return cmp_mix(array_list, dts, array_index + 1, l, r)
    if (dtype & dtypes.STRING_TYPE_OFFSET_MAST) == dtypes.ARRAY_TYPE_STRING:
        str_length = dtype >> dtypes.STRING_TYPE_OFFSET_BITS
        array_str = array.view(np.uint32)
        for i in range(str_offset, str_length):
            val_l = array_str[i + l * str_length]
            val_r = array_str[i + r * str_length]
            if val_l < val_r:
                return -1
            if val_l > val_r:
                return 1
            if val_l == 0:
                break
        return cmp_mix(array_list, dts, array_index + 1, l, r)


@njit(cache=NUMBA_CACHE)
def insertion_argsort0(array_list: List[np.ndarray], array_type_list: List[int], array_index: int,
                       indexes: np.ndarray, array_offset: int, array_length: int, str_offset=0):
    for i in range(array_offset + 1, array_offset + array_length):
        val = indexes[i]
        j = i
        while j > array_offset and cmp_mix(array_list, array_type_list, array_index,
                                           val, indexes[j - 1], str_offset) < 0:
            indexes[j] = indexes[j - 1]
            j -= 1
        indexes[j] = val


@njit(cache=NUMBA_CACHE)
def radix_argsort0_mix(array_list: List[np.ndarray], array_type_list: List[int], array_index: int,
                       indexes: np.ndarray, array_offset: int, array_length: int):
    if array_index >= len(array_list):
        return
    if array_length < INSERTION_SORT_LIMIT:
        return insertion_argsort0(array_list, array_type_list, array_index, indexes, array_offset, array_length)
    array = array_list[array_index]
    dtype = array_type_list[array_index]
    if dtype == dtypes.ARRAY_TYPE_INT64:
        return radix_argsort0_int(array_list, array_type_list, array_index, array.view(np.int64),
                                  indexes, array_offset, array_length)
    if dtype == dtypes.ARRAY_TYPE_FLOAT64:
        return None
    if dtype == dtypes.ARRAY_TYPE_STRING:
        return None


# def get_array_type(array: np.array):
#     if array.dtype.type in [np.int64, np.int32]:
#         return ARRAY_TYPE_INT
#     if array.dtype.type in [np.float32]:
#         return ARRAY_TYPE_FLOAT32
#     if array.dtype.type in [np.float64]:
#         return ARRAY_TYPE_FLOAT64
#     if array.dtype.type in [np.str_]:
#         return ARRAY_TYPE_STRING
#     raise Exception('unsupported dtype: ' + array.dtype)


# def radix_argsort_mix(array_list: List[np.ndarray], indexes: np.ndarray = None,
#                       array_offset: int = 0, array_length: int = 0, array_index: int = 0):
#     if not array_length:
#         array_length = len(array_list[0])
#     if indexes is None:
#         indexes = np.arange(array_length)
#     radix_argsort0_mix(array_list, array_index, [get_array_type(a) for a in array_list],
#                        indexes, array_offset, array_length)
#     return indexes


def radix_argsort(array: np.ndarray, indexes: np.ndarray, unicode=True):
    if not indexes:
        indexes = np.arange(array.shape[0], dtype=int)
    if array.dtype in [int]:
        radix_argsort0_int(array, indexes, 0, len(array))
    elif array.dtype.type in [np.str_]:
        radix_argsort0_str(array.view(np.uint8 if unicode else np.uint32)
                           .reshape(-1, array.itemsize if unicode else array.itemsize // 4), indexes, 0,
                           len(array), 0, unicode=unicode)
    elif array.dtype.type in [np.float32, np.float64]:
        radix_argsort0_float(array, array.view(np.uint64 if array.itemsize == 8 else np.uint32),
                             indexes, 0, len(array), array.itemsize * 8)
    else:
        raise Exception('unsupported data type %s' % array.dtype)
    return indexes
