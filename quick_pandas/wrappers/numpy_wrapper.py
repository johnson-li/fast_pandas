import logging

import numpy as np

from quick_pandas.sort import radix_argsort, radix_sort
from quick_pandas.wrappers.wrapper import Wrapper, FunctionWrapper


class ndarray_wrapper(Wrapper):
    __wrapped_class__ = np.ndarray
    __class__ = np.ndarray

    def __init__(self, *args, **kwargs):
        super(ndarray_wrapper, self).__init__(*args, **kwargs)
        self.__actual_class__ = ndarray_wrapper

    def sort(self, axis=-1, kind='quicksort', order=None):
        if kind == 'radixsort' and len(self.__wrapped_instance__.shape) == 1:
            return radix_sort(self.__wrapped_instance__)
        return self.__wrapped_instance__.sort(axis, kind, order)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        if kind == 'radixsort' and len(self.__wrapped_instance__.shape) == 1:
            return radix_argsort(self.__wrapped_instance__)
        return self.__wrapped_instance__.sort(axis, kind, order)

    def __len__(self):
        return self.__wrapped_instance__.__len__()

    def __getitem__(self, item):
        return self.__wrapped_instance__[item]

    def __setitem__(self, key, value):
        self.__wrapped_instance__[key] = value

    def __eq__(self, other):
        return self.__wrapped_instance__.__eq__(other)

    def __gt__(self, other):
        return self.__wrapped_instance__.__gt__(other)

    def __ge__(self, other):
        return self.__wrapped_instance__.__ge__(other)

    def __lt__(self, other):
        return self.__wrapped_instance__.__lt__(other)

    def __le__(self, other):
        return self.__wrapped_instance__.__le__(other)


class NdarrayFunctionWrapper(FunctionWrapper):
    def __call__(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        if isinstance(res, ndarray_wrapper.__wrapped_class__):
            return ndarray_wrapper(__wrapped_instance__=res)
        else:
            logging.warning("The result %s is not an instance of numpy.ndarray")
            return res
