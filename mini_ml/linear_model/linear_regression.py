from mini_ml.linear_model import LinearModel
from mini_ml.metrics import sum_square_error
import numpy as np

class LinearRegression(LinearModel):
    def __init__(self):
        super(LinearRegression, self).__init__()
    
    def fit(self, X, y):
        self._setup_input(X, y)
        self.X = self._add_intercept(self.X)

        if self.theta == None:
            self.theta = np.zeros(self.X.shape[1], dtype=np.float64)

        for idx in range(self.max_iter):
            y_predict = self.X @ self.theta
            # cost = sum_square_error(y, y_predict)
            delta = np.sum((y - y_predict) * self.X, axis=0)
            self.theta = self.theta + self.step_size * delta
    
    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = self._add_intercept(X)

        return X @ self.theta

    def _add_intercept(self, X):
        """Add intercept to matrix X

        Args:
            x: 2D NumPy array, (n_sample, n_feature)

        Returns:
            New matrix same as x with 1's in the 0th column.
        """
        new_X = np.zeros((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        new_X[:, 0] = 1
        new_X[:, 1:] = X

        return new_X