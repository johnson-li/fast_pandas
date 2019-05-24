import logging

from pandas.core.sorting import *
from pandas import DataFrame

from quick_pandas import sort_api

logger = logging.getLogger('pandas_wrapper')


def argsort(array: np.ndarray, kind):
    if array.dtype.type in [np.int64, np.int32, np.str_, np.float32, np.float64] and kind == 'radixsort':
        return sort_api.radix_argsort(array)
    logger.warning('revert to quick sort')
    return array.argsort(kind='quicksort' if kind == 'radixsort' else kind)


def pandas_core_sorting_nargsort(items, kind='quicksort', ascending=True, na_position='last'):
    """
    This is intended to be a drop-in replacement for np.argsort which
    handles NaNs. It adds ascending and na_position parameters.
    GH #6399, #5231
    """

    # specially handle Categorical
    if is_categorical_dtype(items):
        if na_position not in {'first', 'last'}:
            raise ValueError('invalid na_position: {!r}'.format(na_position))

        mask = isna(items)
        cnt_null = mask.sum()
        sorted_idx = items.argsort(ascending=ascending, kind=kind)
        if ascending and na_position == 'last':
            # NaN is coded as -1 and is listed in front after sorting
            sorted_idx = np.roll(sorted_idx, -cnt_null)
        elif not ascending and na_position == 'first':
            # NaN is coded as -1 and is listed in the end after sorting
            sorted_idx = np.roll(sorted_idx, cnt_null)
        return sorted_idx

    with warnings.catch_warnings():
        # https://github.com/pandas-dev/pandas/issues/25439
        # can be removed once ExtensionArrays are properly handled by nargsort
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message="Converting timezone-aware DatetimeArray to")
        items = np.asanyarray(items)
    idx = np.arange(len(items))
    mask = isna(items)
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]
    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    import time
    ts = time.time()
    # indexes = non_nans.argsort(kind=kind)
    indexes = argsort(non_nans, kind)
    print(type(non_nans))
    print('np argsort takes %fs on %s with type %s' % (time.time() - ts, non_nans.shape, non_nans.dtype))
    ts = time.time()
    indexer = non_nan_idx[indexes]
    print('getting indexer takes %fs on %s' % (time.time() - ts, non_nans.shape))
    if not ascending:
        indexer = indexer[::-1]
    # Finally, place the NaNs at the end or the beginning according to
    # na_position
    if na_position == 'last':
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == 'first':
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError('invalid na_position: {!r}'.format(na_position))
    return indexer


def pandas_read_csv(func):
    def wrapped(*args, **kwargs):
        path = None
        try:
            import datatable as dt
            if len(args) == 1 and len(kwargs) == 0:
                path = args[0]
            elif len(args) == 0 and len(kwargs) == 1 and 'path' in kwargs:
                path = kwargs.pop('path')
        except ImportError as e:
            logger.warning(e)
        if path:
            df = dt.fread(path)
            data = df.to_numpy().T
            return DataFrame(dict(zip(df.names, data)))
        return func(*args, **kwargs)

    return wrapped
