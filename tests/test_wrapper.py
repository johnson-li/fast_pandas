from unittest import TestCase

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


T = T_Wraped


class TestPatch(TestCase):

    def test_wrapper(self):
        t = T()
        self.assertEqual(t.t(), 'tw')
        self.assertEqual(t.tt(), 't')
        t = T('asdf')
        self.assertEqual(t.t(), 'tw')
        self.assertEqual(t.tt(), 'asdf')
