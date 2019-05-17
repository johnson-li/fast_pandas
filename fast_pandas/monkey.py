import numpy as np

from fast_pandas.wrappers.numpy_wrapper import ndarray_wrapper, NdarrayFunctionWrapper


def patch_numpy():
    np.ndarray = ndarray_wrapper
    np.empty = NdarrayFunctionWrapper(np.empty)


def patch_pandas():
    pass


def patch_all():
    patch_numpy()
    patch_pandas()
