from unittest import TestCase

from quick_pandas import monkey

monkey.patch_all()

from quick_pandas.wrappers.numpy_wrapper import ndarray_wrapper


class TestPandas(TestCase):
    def test_dataframe_ndarray(self):
        import pandas as pd
        import numpy as np
        data = np.random.randint(0, 10000, 1000)
        df = pd.DataFrame(data)
        # self.assertEqual(getattr(df._data.blocks[0].values, '__actual_class__', None), ndarray_wrapper)

    def test_dataframe_list(self):
        import pandas as pd
        data = [1, 2, 3]
        df = pd.DataFrame(data)
        # self.assertEqual(getattr(df._data.blocks[0].values, '__actual_class__', None), ndarray_wrapper)
