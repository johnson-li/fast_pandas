from unittest import TestCase
from ext import group_by

from quick_pandas.wrappers.pandas.groupby import *

SIZE = 1000000
RANGE = 100


class TestGroupBy(TestCase):
    def small_df(self):
        a = np.array([1, 1, 3, 3, 3, 1])
        b = np.array([1, 1, 2, 2, 1, 1])
        c = np.array([11, 22, 33, 44, 55, 66])
        d = np.array([111, 222, 333, 444, 555, 666])
        return pd.DataFrame({'A': a, 'B': b, 'C': c, 'D': d})

    def large_df(self):
        return pd.DataFrame({'A': np.random.randint(0, RANGE, SIZE),
                             'B': np.random.randint(0, RANGE, SIZE),
                             'C': np.random.rand(SIZE),
                             'D': np.random.rand(SIZE)})

    def test_numba(self):
        df = self.small_df()
        res = df.groupby(by=['A'], sort=True).transform(np.mean)
        res['A'] = df['A']
        res['B'] = df['B']
        print(res)
        a, b = [df[c].values for c in ['A', 'B']]
        c, d = [df[c].values for c in ['C', 'D']]
        indexes = [a, b]
        res = group_and_transform0(indexes, [c, d])
        print(res)
        print([c, d])

    def test_group(self):
        data = [np.array([1, 2, 1, 3, 4]),
                np.array(['a', 'b', 'a', 'a', 'c'])]
        au8, dts = dtypes.convert_to_uint8(data)
        groups = group(au8, dts, 0, list(np.arange(len(data[0]))))
        print(groups)

    def test_group_by(self):
        df = self.small_df()
        res = df.groupby(by=['A'], sort=True).transform(np.mean)
        print(res)
        res = group_and_transform(df, ['A', 'B'])
        print(res)

    def test_group_by_ext(self):
        data = [np.array([1, 2, 1, 3, 4]),
                np.array(['a', 'b', 'a', 'a', 'c'])]
        res = group_by.group_and_transform(data)
