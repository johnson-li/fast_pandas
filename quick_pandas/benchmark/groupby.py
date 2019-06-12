import time

import numpy as np
import pandas as pd

from quick_pandas.wrappers.pandas.groupby import group_and_transform

RANGE = 10
SIZE = 100000
REPEAT = 5


def group_by():
    df = pd.DataFrame({'A': np.random.randint(0, RANGE, SIZE),
                       'B': np.random.randint(0, 2, SIZE),
                       'C': np.random.rand(SIZE),
                       'D': np.random.rand(SIZE),
                       'E': np.random.randint(0, 2, SIZE).astype(str),
                       })
    by = ['A', 'B']
    targets = ['D']
    for i in range(REPEAT):
        ts = time.time()
        df.groupby(by=by, sort=True).transform(np.mean)
        print('Pandas group and transform takes %fs' % (time.time() - ts))
        ts = time.time()
        group_and_transform(df, by, targets, np.mean)
        print('quick-pandas group and transform takes %fs' % (time.time() - ts))


if __name__ == '__main__':
    group_by()
