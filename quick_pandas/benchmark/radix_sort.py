import logging
import time

import numpy as np

from quick_pandas.sort_api import radix_sort, radix_argsort


def radix_sort_str_benchmark():
    array_range = 10000
    array_length = 10000000
    array = np.random.randint(0, array_range, (array_length,))
    array = array.astype(str)
    radix_argsort(np.zeros((1,), dtype=str))
    radix_argsort(np.zeros((1,), dtype=str))
    for i in range(3):
        ts = time.time()
        np.argsort(array, kind='quicksort')
        logging.warning("[str] quick argsort causes %fs" % (time.time() - ts))
        ts = time.time()
        radix_argsort(array)
        logging.warning("[str] radix argsort(unicode=True) causes %fs" % (time.time() - ts))
        ts = time.time()
        radix_argsort(array, unicode=False)
        logging.warning("[str] radix argsort(unicode=False) causes %fs" % (time.time() - ts))


def radix_sort_float_benchmark():
    array_range = 10000
    array_length = 10000000
    array = np.random.rand(array_length)
    array -= 0.5
    array *= array_range
    for dtype in [np.float64, np.float32]:
        array = array.astype(dtype)
        np.argsort(np.zeros((1,), dtype=dtype))
        radix_argsort(np.zeros((1,), dtype=dtype))
        for i in range(3):
            ts = time.time()
            np.argsort(array, kind='quicksort')
            logging.warning("[%s] quick argsort causes %fs" % (dtype, time.time() - ts))
            ts = time.time()
            radix_argsort(array)
            logging.warning("[%s] radix argsort causes %fs" % (dtype, time.time() - ts))


def radix_sort_int_benchmark():
    array_range = 10000
    array_length = 100000000
    array = np.random.randint(-array_range, array_range, (array_length,))
    radix_sort(np.zeros((1,), dtype=int))
    radix_argsort(np.zeros((1,), dtype=int))
    for i in range(3):
        ts = time.time()
        np.argsort(array, kind='quicksort')
        logging.warning("[int] quick argsort causes %fs" % (time.time() - ts))
        ts = time.time()
        radix_argsort(array)
        logging.warning("[int] radix argsort causes %fs" % (time.time() - ts))
        ts = time.time()
        radix_sort(array)
        logging.warning("[int] radix sort causes %fs" % (time.time() - ts))


if __name__ == '__main__':
    radix_sort_int_benchmark()
    radix_sort_str_benchmark()
    radix_sort_float_benchmark()
