# cython: language_level=3
from typing import List

import numpy as np
cimport cython


def say_hello_to(name):
    print("Hello %s!" % name)

def radix_argsort(array: np.ndarray):
    pass

def group(keys_list: List[np.ndarray], keys_dtype: List[int], keys_index: int, indexes: List[int]):
    pass
