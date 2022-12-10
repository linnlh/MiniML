class LinearModel(object):
    """Base class for linear model"""

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta=None):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X, y):
        """Run solver to fit linear model.

        Args:
            X: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        raise NotImplementedError('Subclass of LinearModel must implement fit method.')
    
    def predict(self, X):
        """Make a prediction given new inputs X.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        raise NotImplementedError('Subclass of LinearModel must implement predict method.')