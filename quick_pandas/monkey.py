import pandas

from quick_pandas import sort_api, config
from quick_pandas.wrappers.pandas_wrapper import pandas_core_sorting_nargsort, pandas_read_csv


def patch_numpy():
    # np.ndarray = ndarray_wrapper
    # np.empty = NdarrayFunctionWrapper(np.empty)
    pass


def patch_pandas():
    pandas.core.sorting.nargsort = pandas_core_sorting_nargsort
    pandas.read_csv = pandas_read_csv(pandas_read_csv)


def init(ascii_sort=False):
    config.ASCII_SORT = ascii_sort
    sort_api.init()


def patch_all():
    init()
    patch_numpy()
    patch_pandas()
