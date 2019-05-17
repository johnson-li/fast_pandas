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
                indexes = radix_argsort(array)
                array_sorted = np.sort(array)
                for i in range(array.shape[0]):
                    self.assertEqual(array[indexes[i]], array_sorted[i])
