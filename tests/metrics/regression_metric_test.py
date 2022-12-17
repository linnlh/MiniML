from mini_ml.metrics import *
import numpy as np

def sum_square_error_test():
    y_true = np.array([3, 2, 5, 9])
    y_predict = np.array([2, 2, 2, 2])

    assert sum_square_error(y_true, y_predict) == 59

if __name__ == "__main__":
    sum_square_error_test()