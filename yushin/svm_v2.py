import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin


class mySVM(BaseEstimator, ClassifierMixin):
    def __init__(self, c=1.15, eta=0.01, forget_factor=0.1, mu = 0.975, v = 0.999,
    max_iter=3000, random_state=1):
        self.c = c
        self.eta = eta
        self.forget_factor = forget_factor
        self.mu = mu
        self.v = v
        self.max_iter = max_iter
        self.random_state = random_state
        
    # Shuffle data
    @staticmethod
    def shuffle(data_size):
        shuffled_indices = np.arange(data_size)
        np.random.shuffle(shuffled_indices)
        return shuffled_indices

    def get_dw(self, x, y, data_size):
        dw = np.zeros(self.w_.shape)
        scores = np.dot(x, self.w_)
        label_scores = scores[np.arange(data_size), y]
        d = scores - label_scores[:, np.newaxis] + 1
        d = np.where(d > 0, 1, d)
        d[np.arange(data_size),y] = 0

        # dw = (1/x.shape[0]) * np.dot(x.T, d)
        dw = (1/x.shape[0]) * np.dot(x.T, d) + (1. / self.c) * self.w_

        return dw

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)

        # Magic number means 10 classes(MNIST)
        self.label_number = 10
        self.data_size = len(y)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(x.shape[1], self.label_number)) 

        # For stabilizing division by zero
        epsilon = 10**(-8)

        # To be parameterized
        batch_size = 128

        n_epochs = batch_size * self.max_iter // self.data_size
        n_batches = self.data_size // batch_size

        # NADAM init
        m = np.zeros_like(self.w_)
        n = np.zeros_like(self.w_)
        mu_pw = self.mu
        v_pw = self.v

        for epoch in range(n_epochs):
            random_indices = mySVM.shuffle(self.data_size)

            for batch in range(n_batches):
                # Pick mini-batch
                batch_indices = random_indices[batch*batch_size:batch*batch_size + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]

                dw = self.get_dw(batch_x, batch_y, batch_size)

                # Naive SGD
                self.w_ -= self.eta*dw

                # # NADAM
                # m = self.mu*m + (1-self.mu)*dw
                # n = self.v*n + (1-self.v)*np.power(dw, 2)
                # m_hat = (self.mu*m/(1 - mu_pw*self.mu)) + ((1-self.mu)*dw/(1-mu_pw))
                # n_hat = self.v*n / (1-v_pw)
                # self.w_ -= self.eta * m_hat / np.sqrt(epsilon + n_hat)

                # mu_pw *= self.mu
                # v_pw *= self.v

        return self
        
    def predict(self, x):
        return np.argmax(np.dot(x, self.w_), axis=1)

    def score(self, x, y):
        err = y - self.predict(x)
        err = np.where(err == 0, 1, 0)
        return sum(err)