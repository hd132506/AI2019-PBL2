from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import random
class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, learning_rate=0.01, max_iter = 1000, batch_size):
        self.C = C
        self.w = np.random.normal(0.0, 0.1, np.shape(X)[1])
        self.b = np.random.normal(0.0, 0.1, 1)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size


    def grad_wb(self, X, y, idx):
        # initialize w, b
        ws = np.zeros(len(x[0]))
        bs = 0.0
        for i in idx :
            if y[i]*(np.dot(w, X[i])+b) < 1:
                ws += -1*y[i]*np.array(X[i]) + 1/self.C*self.w
                bs += -1*y[i] + 0
            else :
                ws += 0 + 1/self.C*self.w
                bs += 0 + 0
        grad_w = ws / self.batch_size
        grad_s = bs / self.batch_size
        return self.w - learning_rate * grad_w, self.b - learning_rate * grad_b

    # def hinge(self, X, y):



    def fit(self, X, y):
        random_idx = range(np.shape(X)[0])
        random.shuffle(random_idx)
        for k in range(self.max_iter):
            batch = random_idx[k*batch_size:(k+1)*batch_size]
            self.w, self.b = grad_wb(X, y, batch)
        return self

    def predict(self, X):
        pred = []
        for img in X :

            pred.append()
        return pred

params = {
    'lr': [0.1, 0.5, 0.7]
}
gs = GridSearchCV(MyClassifier(), param_grid=params, cv=4)

x = np.arange(30)
y = np.concatenate((np.zeros(10), np.ones(10), np.ones(10) * 2))
gs.fit(x, y)
