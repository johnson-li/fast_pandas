from unittest import TestCase

import numpy as np

from fast_pandas.wrappers.numpy_wrapper import ndarray_wrapper, NdarrayFunctionWrapper
from fast_pandas.wrappers.wrapper import Wrapper


class T:
    def __init__(self, a=None):
        self.a = a

    def t(self):
        return self.a if self.a else 't'

    def tt(self):
        return self.t()


class T_Wraped(Wrapper):
    __wrapped_class__ = T

    def t(self):
        return 'tw'


Origin = T
T = T_Wraped


class TestPatch(TestCase):

    def test_wrapper(self):
        t = T()
        self.assertEqual(t.t(), 'tw')
        self.assertEqual(t.tt(), 't')
        t = T('asdf')
        self.assertEqual(t.t(), 'tw')
        self.assertEqual(t.tt(), 'asdf')

    def test_wrapper_agent(self):
        t = Origin('asdf')
        t = T(__wrapped_instance__=t)
        self.assertEqual(t.t(), 'tw')
        self.assertEqual(t.tt(), 'asdf')

    def test_ndarray_wrapper(self):
        np.ndarray = ndarray_wrapper
        np.empty = NdarrayFunctionWrapper(np.empty)
        a = np.empty((1,))
        self.assertEqual(type(a), ndarray_wrapper)
