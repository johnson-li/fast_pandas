import time

import numpy as np
import pandas as pd
from quick_pandas.api.pandas import sort_values

RANGE = 10000000
LENGTH = 10000000


def sort_multi_array():
    a1 = np.random.randint(0, RANGE, LENGTH)
    a2 = np.random.randint(0, RANGE, LENGTH)
    a3 = np.random.randint(0, RANGE, LENGTH)
    ts = time.time()
    df = pd.DataFrame({'a': a1, 'b': a2, 'c': a3})
    df.sort_values(by=['a', 'b', 'c'])
    print('[multi] quick sort takes %fs' % (time.time() - ts))
    ts = time.time()
    sort_values(df, by=['a', 'b', 'c'])
    print('[multi] radix sort takes %fs' % (time.time() - ts))


def sort_single_array():
    a = np.random.randint(0, RANGE, LENGTH)
    df1 = pd.DataFrame({'a': a})
    ts = time.time()
    df1.sort_values(by='a')
    print('[single] quick sort takes %fs' % (time.time() - ts))
    ts = time.time()
    sort_values(df1, by='a')
    print('[single] radix sort takes %fs' % (time.time() - ts))


if __name__ == '__main__':
    for i in range(3):
        sort_single_array()
    for i in range(3):
        sort_multi_array()
