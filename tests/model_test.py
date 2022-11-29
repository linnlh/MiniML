from mini_ml.neighbors import KNNClassifier
from mini_ml.naive_bayes import MultinomialNB

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

if __name__ == "__main__":
    iris = load_iris()

    # split the data to training set and testing set (7:3)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

    clf = KNNClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    precision = (float)(np.sum(y_pred == y_test)) / y_test.shape[0]
    print("precision: ", precision)