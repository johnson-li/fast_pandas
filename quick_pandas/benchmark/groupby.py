import time

import numpy as np
import pandas as pd

from quick_pandas.api.pandas import group_and_transform, radix_argsort_py

RANGE = 1000000000
SIZE = 7000000
REPEAT = 1


def group_by():
    df = pd.DataFrame({'A': np.random.randint(-RANGE, RANGE, SIZE),
                       'D': np.random.rand(SIZE)})
    by = ['A']
    targets = ['D']
    print('dataframe is ready')
    for i in range(REPEAT):
        ts = time.time()
        df.groupby(by=by, sort=False)['D'].transform(np.mean)
        print('Pandas group and transform takes %fs' % (time.time() - ts))
        ts = time.time()
        radix_argsort_py([df['A'].values]) 
        print('quick-pandas sort takes %fs' % (time.time() - ts))
        ts = time.time()
        group_and_transform(df, by, targets, np.mean)
        print('quick-pandas group and transform takes %fs' % (time.time() - ts))


if __name__ == '__main__':
    group_by()
