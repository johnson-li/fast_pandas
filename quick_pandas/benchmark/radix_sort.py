import logging
import time

import numpy as np

from quick_pandas.ext import argsort

RANGE = 100
LENGTH = 100000000


def radix_sort_str_benchmark():
    array_range = RANGE
    array_length = LENGTH
    array = np.random.randint(0, array_range, (array_length,))
    array = array.astype(str)
    for i in range(3):
        ts = time.time()
        np.argsort(array, kind='quicksort')
        logging.warning("[str] quick argsort causes %fs" % (time.time() - ts))
        ts = time.time()
        argsort.radix_argsort_py([array])
        logging.warning("[str] radix argsort causes %fs" % (time.time() - ts))


def radix_sort_float_benchmark():
    array_range = RANGE
    array_length = LENGTH
    array = np.random.rand(array_length)
    array -= 0.5
    array *= array_range
    for dtype in [np.float64]:
        array = array.astype(dtype)
        for i in range(3):
            ts = time.time()
            np.argsort(array, kind='quicksort')
            logging.warning("[%s] quick argsort causes %fs" % (dtype, time.time() - ts))
            ts = time.time()
            argsort.radix_argsort_py([array])
            logging.warning("[%s] radix argsort causes %fs" % (dtype, time.time() - ts))


def radix_sort_int_benchmark():
    array_range = RANGE
    array_length = LENGTH
    array = np.random.randint(-array_range, array_range, (array_length,))
    for i in range(3):
        ts = time.time()
        np.argsort(array, kind='quicksort')
        logging.warning("[int] quick argsort causes %fs" % (time.time() - ts))
        ts = time.time()
        argsort.radix_argsort_py([array])
        logging.warning("[int] radix argsort causes %fs" % (time.time() - ts))


if __name__ == '__main__':
    radix_sort_int_benchmark()
    radix_sort_float_benchmark()
    radix_sort_str_benchmark()
