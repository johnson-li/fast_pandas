from unittest import TestCase

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

    def test_group_by(self):
        df = self.small_df()
        res = df.groupby(by=['A'], sort=True).transform(np.mean)
        print(res)
        res = group_and_transform(df, ['A', 'B'])
        print(res)
