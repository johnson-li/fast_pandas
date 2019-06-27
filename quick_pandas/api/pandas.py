from quick_pandas.ext.group_by import group_and_transform as group_and_transform0
from quick_pandas.ext.argsort import radix_argsort_py
from typing import List, Callable
import pandas as pd


def sort_values(df: pd.DataFrame, by: List[str]):
    if not isinstance(by, (list, tuple)):
        by = []
    columns = [df[d].values for d in by]
    indexes = radix_argsort_py(columns)
    return pd.DataFrame({df[c].values[indexes] for c in df.columns.values})


def group_and_transform(df: pd.DataFrame, by: List[str], targets: List[str], func: Callable):
    if not isinstance(by, (list, tuple)):
        by = [by]
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    res = group_and_transform0(df, by, targets, func)
    return pd.DataFrame({
        **{d: df[d].values for d in by},
        **dict(zip(targets, res))
    })
