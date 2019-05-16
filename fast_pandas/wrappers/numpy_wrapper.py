import numpy as np

from fast_pandas.wrappers.wrapper import Wrapper


class ndarray_wrapper(Wrapper):
    __wrapped_class__ = np.ndarray
