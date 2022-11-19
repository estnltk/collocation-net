from numpy import diag
from pandas import DataFrame

from typing import List


def get_confusion_matrix(tbl: DataFrame, assignment_columns: List[str]) -> DataFrame:
    """Creates confusion matrix out of label assignments stored as dataframe columns"""
    assert len(assignment_columns) == 2, "There can be only two assignment columns"
    return DataFrame(tbl.value_counts(assignment_columns)).unstack().droplevel(0, axis=1)

def observed_agreement(confusion_matrix: DataFrame) -> float:
    """Returns obsered agreement in the range [0,1]."""
    return diag(confusion_matrix).sum()/confusion_matrix.sum().sum()
