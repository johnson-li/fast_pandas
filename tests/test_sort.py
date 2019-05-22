from unittest import TestCase

import numpy as np

from quick_pandas.sort_api import radix_sort, radix_argsort


class TestSort(TestCase):
    def test_radix_sort(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.randint(-array_range, array_range, (array_length,))
                radix_sort(array)
                for i in range(1, len(array)):
                    self.assertGreaterEqual(array[i], array[i - 1])

    def test_radix_argsort(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.randint(-array_range, array_range, (array_length,))
                array_cpy = array.copy()
                indexes = radix_argsort(array)
                array_sorted = np.sort(array)
                for i in range(array.shape[0]):
                    self.assertEqual(array_cpy[indexes[i]], array_sorted[i])

    def test_radix_argsort_str(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000, 1000000]:
                array = np.random.randint(0, array_range, (array_length,))
                array = array.astype(str)
                array_cpy = array.copy()
                indexes = radix_argsort(array)
                array_sorted = np.sort(array_cpy)
                for i in range(array.shape[0]):
                    self.assertEqual(array_cpy[indexes[i]], array_sorted[i])

    def test_radix_argsort_str_quick(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.randint(0, array_range, (array_length,))
                array = array.astype(str)
                array_cpy = array.copy()
                indexes = radix_argsort(array, unicode=False)
                array_sorted = np.sort(array_cpy)
                for i in range(array.shape[0]):
                    self.assertEqual(array_cpy[indexes[i]], array_sorted[i])

    def test_radix_argsort_float64(self):
        np.set_printoptions(formatter={'int': hex})
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            for array_length in [10, 100, 1000, 10000, 100000]:
                array = np.random.rand(array_length)
                array -= 0.5
                array *= array_range
                array[0] = np.nan
                array[1] = np.inf
                array[2] = -np.inf
                array_cpy = array.copy()
                indexes = radix_argsort(array)
                array_sorted = np.sort(array_cpy)
                for i in range(array.shape[0]):
                    if np.isnan(array_sorted[i]):
                        self.assertTrue(np.isnan(array_cpy[indexes[i]]))
                    else:
                        self.assertEqual(array_cpy[indexes[i]], array_sorted[i])

    def test_radix_argsort_float32(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.rand(array_length)
                array = array.astype(np.float32)
                array -= 0.5
                array *= array_range
                array_cpy = array.copy()
                indexes = radix_argsort(array)
                array_sorted = np.sort(array_cpy)
                for i in range(array.shape[0]):
                    if np.isnan(array_sorted[i]):
                        self.assertTrue(np.isnan(array_cpy[indexes[i]]))
                    else:
                        self.assertEqual(array_cpy[indexes[i]], array_sorted[i])
