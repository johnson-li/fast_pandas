import numpy as np

NP_FUNCS = [np.mean, np.sum, np.median]

NP_FUNCS_MEAN = 0
NP_FUNCS_SUM = 1
NP_FUNCS_MEDIAN = 2

NP_FUNCS_MAP = dict(zip(range(len(NP_FUNCS)), NP_FUNCS))
NP_FUNCS_MAP_REVERSE = {val: key for key, val in NP_FUNCS_MAP.items()}
