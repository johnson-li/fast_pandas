from unittest import TestCase

from ext import group_by
from quick_pandas.wrappers.pandas.groupby import *

SIZE = 1000
RANGE = 10


class TestGroupBy(TestCase):
    def small_df(self):
        a = np.array([1, 1, 3, 3, 3, 1])
        b = np.array([1, 1, 2, 2, 10, 1])
        c = np.array([11, 22, 33, 44, 55, 66])
        d = np.array([111, 222, 333, 444, 555, 666])
        return pd.DataFrame({'A': a, 'B': b, 'C': c, 'D': d})

    def large_df(self):
        return pd.DataFrame({'A': np.random.randint(0, RANGE, SIZE),
                             'B': np.random.randint(0, RANGE, SIZE),
                             'C': np.random.rand(SIZE),
                             'D': np.random.rand(SIZE),
                             'E': np.random.randint(0, 2, SIZE).astype(str),
                             })

    def test_group(self):
        data = [np.array([1, 2, 1, 3, 4]),
                np.array(['a', 'b', 'a', 'a', 'c'])]
        au8, dts = dtypes.convert_to_uint8(data)
        groups = group(au8, dts, 0, list(np.arange(len(data[0]))))

    def test_group_by(self):
        df1 = self.large_df()
        df2 = df1.copy()
        by = ['A', 'B', 'E']
        targets = ['C', 'D']
        res1 = df1.groupby(by=by, sort=True).transform(np.mean)
        for b in by:
            res1[b] = df1[b]
        res2 = group_and_transform(df2, by, targets, np.mean, inplace=False, sort=False)
        for t in targets:
            self.assertTrue((res1[t].values == res2[t].values).all())

    def test_group_by_ext(self):
        data = [np.array([1, 2, 1, 3, 4]),
                np.array(['a', 'b', 'a', 'a', 'c'])]
        # res = group_by.group_and_transform(data)
