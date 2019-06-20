# cython: language_level=3

from quick_pandas.ext.operators cimport OPERATOR_INT, OPERATOR_LONG, OPERATOR_FLOAT, OPERATOR_DOUBLE, sum_int, sum_long, sum_float, sum_double, mean_int, mean_long, mean_float, mean_double

cdef group_and_transform0(df: pd.DataFrame, by_columns: List[str], targets: List[str], func: Callable,
                          OPERATOR_INT op_int, OPERATOR_LONG op_long, 
                          OPERATOR_FLOAT op_float, OPERATOR_DOUBLE op_double, sort, inplace)
