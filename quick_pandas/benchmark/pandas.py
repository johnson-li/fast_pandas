import time

import numpy as np
import pandas as pd

from quick_pandas import monkey

monkey.patch_all()


def sort():
    data1 = np.random.randint(0, 10000, 10000000)
    data2 = data1.copy()
    df1 = pd.DataFrame(data=data1)
    df2 = pd.DataFrame(data=data2)
    ts = time.time()
    df1.sort_values(kind='quicksort', by=0)
    print('quick sort takes %fs' % (time.time() - ts))
    ts = time.time()
    df2.sort_values(kind='radixsort', by=0)
    print('radix sort takes %fs' % (time.time() - ts))


if __name__ == '__main__':
    sort()
