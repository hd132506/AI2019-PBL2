import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin



class BinarySVM:
    def __init__(self, c=1.15, eta=0.01, forget_factor=0.1, max_iter=2000, random_state=1, label=-1):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.c = c
        self.forget_factor = forget_factor
        # The label it classifies (+/-)
        self.label = label
    
    # Shuffle data
    @staticmethod
    def shuffle(data_size):
        shuffled_indices = np.arange(data_size)
        np.random.shuffle(shuffled_indices)
        return shuffled_indices

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.data_size = len(y)

        # For stabilizing division by zero
        epsilon = 10**(-6)

        # To be parameterized
        batch_size = 128

        n_epochs = batch_size * self.max_iter // self.data_size
        n_batches = self.data_size // batch_size

        for epoch in range(n_epochs):
            random_indices = BinarySVM.shuffle(self.data_size)

            # RMS init
            rms = 0

            for batch in range(n_batches):
                # Pick mini-batch
                batch_indices = random_indices[batch*batch_size:batch*batch_size + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]

                # Calculate dw
                dw = np.zeros(self.w_.shape)
                for i, _ in enumerate(batch_y):
                    if batch_y[i] * self.hypothesis(batch_x[i]) < 1:
                        dw[1:] -= batch_y[i] * batch_x[i]
                        dw[0] -= batch_y[i]
                dw[1:] = dw[1:] / batch_size + (1. / self.c) * self.w_[1:]
                dw[0] /= batch_size

                # Update
                # Naive SGD : self.w_ -= self.eta * dw 
                # RMS update
                rms = self.forget_factor * rms + (1 - self.forget_factor) * (dw**2)
                self.w_ -= self.eta * dw / np.sqrt(epsilon + rms)

            # print('epoch', epoch, self.cost(x, y, self.data_size))
        return self

    def predict(self, x):
        print(self.hypothesis(x))
        return np.where(self.hypothesis(x) >= 0.0, 1, -1)


    def cost(self, x, y, data_size):
        return (self.c*(self.w_**2)).sum()/2.0 + np.maximum((1 - y * self.hypothesis(x)), 0).sum() / data_size

    def hypothesis(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]


class mySVM(BaseEstimator, ClassifierMixin):
    def __init__(self, c=1.15, eta=0.01, forget_factor=0.1, max_iter=1000, random_state=1):
        self.c = c
        self.eta = eta
        self.forget_factor = forget_factor
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, x, y):
        self.bin_classifiers = [BinarySVM(c= self.c, eta=self.eta, forget_factor=self.forget_factor,
        max_iter=self.max_iter, random_state=self.random_state, label=label) \
            for label in np.unique(y)]

        for classifier in self.bin_classifiers:
            classifier.fit(x, np.where(y == classifier.label, 1, -1))
        return self
        
    def predict(self, x):
        prediction = np.argmax([classifier.hypothesis(x) for classifier in self.bin_classifiers], axis=0)
        prediction = np.array([self.bin_classifiers[p].label for p in prediction])
        return prediction

    def score(self, x, y):
        err = y - self.predict(x)
        err = np.where(err == 0, 1, 0)
        return sum(err)