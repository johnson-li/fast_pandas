from unittest import TestCase

import quick_pandas.sort
from quick_pandas.sort import *


class TestSort(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSort, self).__init__(*args, **kwargs)

    # def test_radix_sort(self):
    #     for array_range in [1, 10, 100, 1000, 10000, 100000]:
    #         for array_length in [1, 10, 100, 1000, 10000, 100000]:
    #             array = np.random.randint(-array_range, array_range, (array_length,))
    #             radix_sort(array)
    #             for i in range(1, len(array)):
    #                 self.assertGreaterEqual(array[i], array[i - 1])

    def test_radix_argsort_int(self):
        quick_pandas.sort.INSERTION_SORT_LIMIT = 0
        for array_range in [1, 10, 100, 1000, 10000, 100000]:
            for array_length in [1, 10, 100, 1000, 10000, 100000]:
                array = np.random.randint(-array_range, array_range, (array_length,))
                au8, dts = dtypes.convert_to_uint8([array])
                indexes = np.arange(array_length)
                ranges = radix_argsort0_int(au8, dts, 0, array, indexes, 0, array_length)
                array_sorted = np.sort(array)
                for i in range(array_length):
                    self.assertEqual(array[indexes[i]], array_sorted[i])
                pre_max = None
                length = 0
                for r in ranges:
                    start, end, array_index, uniform = r
                    length += end - start
                    max_val = array[indexes[start]]
                    for i in range(start, end):
                        if uniform:
                            self.assertEqual(array[indexes[i]], array[indexes[start]])
                        if pre_max is not None:
                            self.assertGreater(array[indexes[i]], pre_max)
                        if array[indexes[i]] > max_val:
                            max_val = array[indexes[i]]
                    pre_max = max_val
                self.assertEqual(array_length, length)

    def test_radix_argsort_str(self):
        quick_pandas.sort.INSERTION_SORT_LIMIT = 0
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            for array_length in [1, 10, 100, 1000, 10000, 10000]:
                array = np.random.randint(0, array_range, (array_length,))
                array = array.astype(str)
                au8, dts = dtypes.convert_to_uint8([array])
                indexes = np.arange(array_length)
                ranges = radix_argsort0_str(au8, dts, 0, array.view(np.uint8), indexes, 0, array_length, array.itemsize)
                array_sorted = np.sort(array)
                for i in range(array.shape[0]):
                    self.assertEqual(array[indexes[i]], array_sorted[i])
                pre_max = None
                length = 0
                for r in ranges:
                    start, end, array_index, uniform = r
                    length += end - start
                    max_val = array[indexes[start]]
                    for i in range(start, end):
                        if uniform:
                            self.assertEqual(array[indexes[i]], array[indexes[start]])
                        if pre_max is not None:
                            self.assertGreater(array[indexes[i]], pre_max)
                        if array[indexes[i]] > max_val:
                            max_val = array[indexes[i]]
                    pre_max = max_val
                self.assertEqual(array_length, length)

    def test_radix_argsort_float64(self):
        quick_pandas.sort.INSERTION_SORT_LIMIT = 0
        for array_range in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            for array_length in [10, 100, 1000, 10000, 10000]:
                array = np.random.rand(array_length)
                array -= 0.5
                array *= array_range
                array[0] = np.nan
                array[1] = np.inf
                array[2] = -np.inf
                au8, dts = dtypes.convert_to_uint8([array])
                indexes = np.arange(array_length)
                ranges = radix_argsort0_float(au8, dts, 0, array.view(np.uint64), indexes, 0, array_length)
                array_sorted = np.sort(array)
                for i in range(array_length):
                    if np.isnan(array_sorted[i]):
                        self.assertTrue(np.isnan(array[indexes[i]]))
                    else:
                        self.assertEqual(array[indexes[i]], array_sorted[i])
                pre_max = None
                length = 0
                for r in ranges:
                    start, end, array_index, uniform = r
                    length += end - start
                    max_val = array[indexes[start]]
                    for i in range(start, end):
                        if uniform:
                            if np.isnan(array[indexes[i]]):
                                self.assertTrue(np.isnan(array[indexes[start]]))
                            else:
                                self.assertEqual(array[indexes[i]], array[indexes[start]])
                        if pre_max is not None:
                            if not np.isnan(array[indexes[i]]):
                                self.assertGreater(array[indexes[i]], pre_max)
                        if array[indexes[i]] > max_val:
                            max_val = array[indexes[i]]
                    pre_max = max_val
                self.assertEqual(array_length, length)

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
                    ranges = insertion_argsort0(au8, dts, 0, indexes, 0, array_length)
                    res = np.argsort(array)
                    for i in range(array_length):
                        a = array[res[i]]
                        b = array[indexes[i]]
                        if dtype in [np.float64, np.float32] and np.isnan(a):
                            self.assertEqual(np.isnan(a), np.isnan(b))
                        else:
                            self.assertEqual(a, b)
                    pre = None
                    length = 0
                    for r in ranges:
                        start, end, array_index, uniform = r
                        for i in range(start + 1, end):
                            self.assertEqual(array[indexes[start]], array[indexes[i]])
                        if pre:
                            self.assertGreater(array[indexes[start]], pre)
                        pre = array[indexes[start]]
                        length += end - start
                    self.assertEqual(array_length, length)

    def test_range(self):
        @njit()
        def outer():
            short_rgs = []
            long_rgs = [(0, 16)]
            while long_rgs:
                rg = long_rgs.pop(0)
                for r in inner(rg):
                    if r[1] - r[0] > 2:
                        long_rgs.append(r)
                    else:
                        short_rgs.append(r)
            return short_rgs

        @njit()
        def inner(rg: List):
            mid = (rg[0] + rg[1]) // 2
            return [(rg[0], mid), (mid, rg[1])]

        self.assertEqual([(i, i + 2) for i in range(0, 16, 2)], outer())

    def test_radix_argsort_mix(self):
        quick_pandas.sort.INSERTION_SORT_LIMIT = 64
        array_size = 1000
        array_range = 100
        array = np.random.randint(0, array_range, array_size)
        int_array = array - array_range // 2
        float_array = array / 10
        str_array = array.astype(str)
        for array in [int_array, float_array, str_array]:
            indexes = np.arange(array_size)
            au8, dts = dtypes.convert_to_uint8([array])
            ranges = radix_argsort0_mix(au8, dts, indexes)
            array_sorted = np.sort(array)
            for i in range(array_size):
                if array.dtype.type in [np.float32, np.float64] and np.isnan(array_sorted[i]):
                    self.assertTrue(np.isnan(array[indexes[i]]))
                else:
                    self.assertEqual(array[indexes[i]], array_sorted[i])
            self.assertEqual(array_size, sum([r[1] - r[0] for r in ranges]))

    def test_radix_argsort_multiarray(self):
        arrays = [np.array(['1', '11', '12', '1', '12']), np.array([1, 11, 12, 1, 12]),
                  np.array([1.1, 2.2, 3.3, 1.1, 1.1])]
        au8, dts = dtypes.convert_to_uint8(arrays)
        indexes = np.arange(5)
        ranges = radix_argsort0_mix(au8, dts, indexes)
        print(ranges)
        for array in arrays:
            print(array[indexes])
        for i in range(4):
            self.assertEqual([(0, 2, 0, False), (2, 3, 0, False), (3, 4, 0, False), (4, 5, 0, False)][i], ranges[i])
        # self.assertEqual(['1', '1', '11', '12', '12'], arrays[0][indexes])
        # self.assertEqual([1, 1, 11, 12, 12], arrays[1][indexes])
        # self.assertEqual([1.1, 1.1, 2.2, 1.1, 3.3], arrays[2][indexes])
