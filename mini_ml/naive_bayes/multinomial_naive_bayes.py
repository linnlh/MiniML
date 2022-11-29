from mini_ml.naive_bayes import _BaseNB

import numpy as np

class MultinomialNB(_BaseNB):
    def __init_(self):
        return
    
    def fit(self, X, y):
        self.class_ = np.unique(y)
        self.n_sample_, self.n_feature_ = X.shape

        n_class_ = self.class_.shape[0]
        self.class_count_ = np.zeros((n_class_), dtype=np.float64)

        self.feature_count_ = np.zeros((n_class_, self.n_feature_), dtype=np.float64)

        for idx, c in enumerate(self.class_):
            X_c = X[y == c]
            
            self.class_count_[idx], _ = X_c.shape
            self.feature_count_[idx, :] = np.sum(X_c, axis=0)
        
        self._update_class_log_prior()
        self._update_feature_log_cond_prob()

    def predict(self, X):
        jll = X @ self.feature_log_prob.T
        jll += self.prior_

        # return self.class_[np.argmax(jll, axis=1)]
        return np.argmax(jll, axis=1)

    def _update_feature_log_cond_prob(self):
        smooth_fc = self.feature_count_ + 1
        smooth_cc = np.sum(smooth_fc, axis=1)

        self.feature_log_prob = np.log(smooth_fc) - np.log(smooth_cc.reshape(-1, 1))

    def _update_class_log_prior(self):
        self.prior_ = np.log(self.class_count_) - np.log(self.n_sample_)