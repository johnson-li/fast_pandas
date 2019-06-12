from unittest import TestCase

import numpy as np

import ext.argsort

RANGE = 100000000
SIZE = 300000


class TestSort(TestCase):
    def test_compare_int(self):
        self.assertEqual(-1, ext.argsort.compare_py([np.array([1, 2])], 0, 1))
        self.assertEqual(1, ext.argsort.compare_py([np.array([2, 1])], 0, 1))
        self.assertEqual(1, ext.argsort.compare_py([np.array([1, 2])], 1, 0))

    def test_compare_float(self):
        self.assertEqual(1, ext.argsort.compare_py([np.array([2, 1], dtype=np.float64)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([1, 2], dtype=np.float64)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([1, np.nan], dtype=np.float64)], 0, 1))
        self.assertEqual(0, ext.argsort.compare_py([np.array([np.nan, np.nan], dtype=np.float64)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([1, np.inf], dtype=np.float64)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([-np.inf, 2], dtype=np.float64)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([-np.inf, np.inf], dtype=np.float64)], 0, 1))
        self.assertEqual(1, ext.argsort.compare_py([np.array([np.nan, np.inf], dtype=np.float64)], 0, 1))
        self.assertEqual(1, ext.argsort.compare_py([np.array([-np.nan, np.inf], dtype=np.float64)], 0, 1))

    def test_compare_string(self):
        self.assertEqual(1, ext.argsort.compare_py([np.array([2, 1], dtype=str)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([1, 2], dtype=str)], 0, 1))
        self.assertEqual(1, ext.argsort.compare_py([np.array(['1', ''], dtype=str)], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array(['112', '113'], dtype=str)], 0, 1))
        self.assertEqual(0, ext.argsort.compare_py([np.array(['112', '112'], dtype=str)], 0, 1))

    def test_compare_mix(self):
        self.assertEqual(-1, ext.argsort.compare_py([np.array([1, 1]), np.array([1, 2])], 0, 1))
        self.assertEqual(-1, ext.argsort.compare_py([np.array([2, 2]), np.array([1.2, 2.1])], 0, 1))
        self.assertEqual(1, ext.argsort.compare_py([np.array([2, 1]), np.array([1.2, 2.1])], 0, 1))

    def test_insertion_argsort(self):
        indexes = ext.argsort.insertion_argsort_py([np.arange(10)])
        self.assertTrue((indexes == np.arange(10)).all())
        indexes = ext.argsort.insertion_argsort_py([np.invert(np.arange(10))])
        self.assertTrue((indexes == np.arange(10)[::-1]).all())
        indexes = ext.argsort.insertion_argsort_py([np.ones(10), np.invert(np.arange(10))])
        self.assertTrue((indexes == np.arange(10)[::-1]).all())
        array = np.random.randint(-RANGE, RANGE, size=10000)
        indexes1 = ext.argsort.insertion_argsort_py([array])
        indexes2 = np.argsort(array)
        for i in range(10000):
            self.assertEqual(array[indexes2[i]], array[indexes1[i]])

    def test_radix_argsort_int(self):
        indexes = ext.argsort.radix_argsort_py([np.arange(10)])
        self.assertTrue((indexes == np.arange(10)).all())
        indexes = ext.argsort.radix_argsort_py([np.invert(np.arange(10))])
        self.assertTrue((indexes == np.arange(10)[::-1]).all())
        indexes = ext.argsort.radix_argsort_py([np.ones(10, dtype=int), np.invert(np.arange(10))])
        self.assertTrue((indexes == np.arange(10)[::-1]).all())
        array = np.random.randint(-RANGE, RANGE, size=SIZE)
        indexes1 = ext.argsort.radix_argsort_py([array])
        indexes2 = np.argsort(array)
        for i in range(SIZE):
            self.assertEqual(array[indexes2[i]], array[indexes1[i]])

    def test_radix_argsort_float(self):
        indexes = ext.argsort.radix_argsort_py([np.arange(10, dtype=float)])
        self.assertTrue((indexes == np.arange(10)).all())
        indexes = ext.argsort.radix_argsort_py([np.arange(10, 0, -1, dtype=float)])
        self.assertTrue((indexes == np.arange(10)[::-1]).all())
        indexes = ext.argsort.radix_argsort_py([np.ones(10, dtype=float), np.arange(10, 0, -1, dtype=float)])
        self.assertTrue((indexes == np.arange(10)[::-1]).all())
        array = np.random.random(SIZE) * RANGE
        indexes1 = ext.argsort.radix_argsort_py([array])
        indexes2 = np.argsort(array)
        for i in range(SIZE):
            self.assertEqual(array[indexes2[i]], array[indexes1[i]])

    def test_radix_argsort_string(self):
        indexes = ext.argsort.radix_argsort_py([np.arange(9).astype(str)])
        self.assertTrue((indexes == np.arange(9)).all())
        indexes = ext.argsort.radix_argsort_py([np.arange(9, 0, -1).astype(str)])
        self.assertTrue((indexes == np.arange(9)[::-1]).all())
        indexes = ext.argsort.radix_argsort_py([np.ones(9).astype(str), np.arange(9, 0, -1).astype(str)])
        self.assertTrue((indexes == np.arange(9)[::-1]).all())
        array = np.random.randint(0, RANGE, size=SIZE).astype(str)
        indexes1 = ext.argsort.radix_argsort_py([array])
        indexes2 = np.argsort(array)
        for i in range(10):
            self.assertEqual(array[indexes2[i]], array[indexes1[i]])

    def test_radix_argsort(self):
        array = [
            np.array(['1', '1', '11', '0000', '0']),
            np.array([3, 2, 1, 4, 5])
        ]
        indexes = ext.argsort.radix_argsort_py(array).tolist()
        self.assertEqual([4, 3, 1, 0, 2], indexes)
