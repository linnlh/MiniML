"""
The :mod:`mini_ml.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""

from .classification_metric import accuracy

from .regression_metric import mean_square_error
from .regression_metric import sum_square_error

__all__ = [
    "accuracy",
    "mean_square_error",
    "sum_square_error"
]