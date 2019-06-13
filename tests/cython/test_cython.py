import time
from unittest import TestCase

import numpy as np

from ext import example


def compute_np(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c


class TestCython(TestCase):
    def test_example(self):
        example.clip(1, 2, 3)
        array_1 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
        array_2 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
        a = 4
        b = 3
        c = 9
        ts = time.time()
        compute_np(array_1, array_2, a, b, c)
        print('%fs' % (time.time() - ts))
        ts = time.time()
        example.compute(array_1, array_2, a, b, c)
        print('%fs' % (time.time() - ts))
