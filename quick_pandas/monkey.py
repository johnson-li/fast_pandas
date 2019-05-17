import pandas

from quick_pandas import sort_api
from quick_pandas.wrappers.pandas_wrapper import pandas_core_sorting_nargsort


def patch_numpy():
    # np.ndarray = ndarray_wrapper
    # np.empty = NdarrayFunctionWrapper(np.empty)
    pass


def patch_pandas():
    pandas.core.sorting.nargsort = pandas_core_sorting_nargsort


def init():
    sort_api.init()


def patch_all():
    init()
    patch_numpy()
    patch_pandas()
