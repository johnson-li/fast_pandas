from unittest import TestCase

import quick_pandas.sort
from quick_pandas.sort import *
from quick_pandas.sort_api import radix_sort, radix_argsort


class TestSort(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSort, self).__init__(*args, **kwargs)
        quick_pandas.sort.INSERTION_SORT_LIMIT = 0

    def test_radix_sort(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.randint(-array_range, array_range, (array_length,))
                radix_sort(array)
                for i in range(1, len(array)):
                    self.assertGreaterEqual(array[i], array[i - 1])

    def test_radix_argsort_int(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.randint(-array_range, array_range, (array_length,))
                au8, dts = dtypes.convert_to_uint8([array])
                indexes = np.arange(array_length)
                radix_argsort0_int(au8, dts, 0, array, indexes, 0, array_length)
                array_sorted = np.sort(array)
                for i in range(array_length):
                    self.assertEqual(array[indexes[i]], array_sorted[i])

    def test_radix_argsort_str(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
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

    def test_cmp_mix_single(self):
        for dtype in [np.int64, np.float64, np.str]:
            array = np.array([121, 12], dtype=dtype)
            arrays = [array]
            arrays_uint8 = [a.view(np.uint8) for a in arrays]
            res = cmp_mix(arrays_uint8, dtypes.get_dtypes(arrays), 0, 0, 1)
            self.assertEqual(res, 1)
            res = cmp_mix(arrays_uint8, dtypes.get_dtypes(arrays), 0, 1, 0)
            self.assertEqual(res, -1)
            res = cmp_mix(arrays_uint8, dtypes.get_dtypes(arrays), 0, 0, 0)
            self.assertEqual(res, 0)

    def test_cmp_mix_multiple(self):
        arrays = [
            np.array([1, 1]),
            np.array([1.1, 1.1]),
            np.array(['1.11', '1.1']),
        ]
        arrays_uint8, dts = dtypes.convert_to_uint8(arrays)
        res = cmp_mix(arrays_uint8, dts, 0, 0, 1)
        self.assertEqual(res, 1)
        res = cmp_mix(arrays_uint8, dts, 0, 1, 0)
        self.assertEqual(res, -1)

    def test_insertion_argsort(self):
        for array_range in [1, 10, 100, 1000, 10000, 100000]:
            for array_length in [1, 10, 100, 1000]:
                for dtype in [np.int64, np.float64, np.str]:
                    array = np.random.randint(-array_range, array_range, (array_length,))
                    array = array.astype(dtype)
                    au8, dts = dtypes.convert_to_uint8([array])
                    indexes = np.arange(array_length)
                    insertion_argsort0(au8, dts, 0, indexes, 0, array_length)
                    res = np.argsort(array)
                    for i in range(array_length):
                        a = array[res[i]]
                        b = array[indexes[i]]
                        if dtype in [np.float64, np.float32] and np.isnan(a):
                            self.assertEqual(np.isnan(a), np.isnan(b))
                        else:
                            self.assertEqual(a, b)

    # def test_radix_argsort_mix(self):
    #         array_size = 1000
    #         array_range = 100
    #         array = np.random.randint(0, array_range, array_size)
    #         int_array = array - array_range // 2
    #         float_array = array / 10
    #         str_array = array.astype(str)
    #         for array in [int_array, float_array, str_array]:
    #             array_cpy = array.copy()
    #             indexes = radix_argsort_mix([array])
    #             array_sorted = np.sort(array)
    #             for i in range(array.shape[0]):
    #                 if array.dtype.type in [np.float32, np.float64] and np.isnan(array_sorted[i]):
    #                     pass
    #                     # self.assertTrue(np.isnan(array_cpy[indexes[i]]))
    #                 else:
    #                     pass
    #                     # self.assertEqual(array_cpy[indexes[i]], array_sorted[i])
