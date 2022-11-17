import numpy as np
from scipy import stats

from mini_ml.distance import euclidean

class KNNClassifier:

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        predictions = [self.__predict_row(x) for x in X]
        
        return predictions

    def __predict_row(self, x):
        distances = [euclidean(x, train_example) for train_example in self.X]
        
        top_k_idx = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        top_k = self.y[top_k_idx]
        
        val, cnts = np.unique(top_k, return_counts=True)
        return val[np.argmax(cnts)]
        