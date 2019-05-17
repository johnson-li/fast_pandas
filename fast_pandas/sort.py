import numpy as np
from numba import njit

INSERTION_SORT_LIMIT = 64
RADIX_BITS = 16


@njit()
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


@njit()
def radix_argsort0(array: np.ndarray, indexes: np.ndarray, array_offset: int, array_length: int):
    if array_length == 0:
        return
    if array_length <= INSERTION_SORT_LIMIT:
        for i in range(array_offset + 1, array_offset + array_length):
            val = array[indexes[i]]
            j = i
            while j > array_offset and val < array[indexes[j - 1]]:
                array[indexes[j]] = array[indexes[j - 1]]
                j -= 1
            array[indexes[j]] = val
        return
    min_val = 0
    max_val = 0
    for i in range(array_offset + 1, array_offset + array_length):
        index = indexes[i]
        if array[index] > max_val:
            max_val = array[index]
        if array[index] < min_val:
            min_val = array[index]
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
    if not last:
        for i in range(1, bin_length + 1):
            radix_argsort0(array, indexes, array_offset + bins[i - 1], bins[i] - bins[i - 1])


def radix_argsort(array: np.ndarray):
    indexes = np.arange(array.shape[0])
    radix_argsort0(array, indexes, 0, len(array))
    return indexes
