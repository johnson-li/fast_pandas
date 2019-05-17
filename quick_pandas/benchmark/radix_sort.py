import logging
import time

import numpy as np

from quick_pandas.sort_api import radix_sort, radix_argsort


def radix_sort_benchmark():
    array_range = 10000
    array_length = 100000000
    array = np.random.randint(-array_range, array_range, (array_length,))
    radix_sort(np.zeros((1,), dtype=int))
    radix_argsort(np.zeros((1,), dtype=int))
    for i in range(1):
        ts = time.time()
        np.sort(array, kind='quicksort')
        logging.warning("quick sort causes %fs" % (time.time() - ts))
        ts = time.time()
        radix_argsort(array)
        logging.warning("radix argsort causes %fs" % (time.time() - ts))
        ts = time.time()
        radix_sort(array)
        logging.warning("radix sort causes %fs" % (time.time() - ts))


if __name__ == '__main__':
    radix_sort_benchmark()
