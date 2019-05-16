
def patch_numpy():
    import numpy as np
    from fast_pandas.sort import radix_sort
    np.sort


def patch_pandas():
    pass


def patch_all():
    patch_numpy()
    patch_pandas()
