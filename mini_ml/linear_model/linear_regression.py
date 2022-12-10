from mini_ml.linear_model import LinearModel
from mini_ml.metrics import mean_square_error
import numpy as np

class LinearRegression(LinearModel):
    def __init__(self):
        super(LinearRegression, self).__init__()
    
    def fit(self, X, y):
        self.X = self._add_intercept(X)
        self.y = y

        if self.theta == None:
            self.theta = np.zeros(X.shape[1], dtype=np.float64)

        for idx in range(self.max_iter):
            y_predict = self.X @ self.theta
    
    def predict(self, X):
        return

    def _add_intercept(X):
        """Add intercept to matrix x.

        Args:
            x: 2D NumPy array, (n_sample, n_feature)

        Returns:
            New matrix same as x with 1's in the 0th column.
        """
        new_X = np.zeros((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        new_X[:, 0] = 1
        new_X[:, 1:] = X

        return new_X