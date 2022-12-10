import numpy as np

def sum_square_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def mean_square_error(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2, axis=0)
