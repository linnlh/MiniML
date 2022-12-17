import numpy as np

class BaseEstimator(object):
    def __init__(self):
        self.y_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Args:
            X: input X matrix (n_sample, n_feature)
            y: target label
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.y_required == True:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
        
        self.X = X
        self.y = y
