import numpy as np
import random



class BinarySVM:
    def __init__(self, c=1.15, eta=0.01, max_iter=2000, random_state=1, label=-1):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.c = c
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


        # To be parameterized
        batch_size = 64

        n_epochs = batch_size * self.max_iter // self.data_size
        n_batches = self.data_size // batch_size

        for _epoch in range(n_epochs):
            random_indices = BinarySVM.shuffle(self.data_size)
            for batch in range(n_batches):
                batch_indices = random_indices[batch*batch_size:batch*batch_size + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                dw = np.zeros(self.w_.shape)

                for i, _ in enumerate(batch_y):
                    if batch_y[i] * self.hypothesis(batch_x[i]) < 1:
                        dw[1:] += -batch_y[i] * batch_x[i]
                        dw[0] += -batch_y[i]
                dw[1:] = dw[1:] / batch_size + self.c * self.w_[1:]
                dw[0] /= batch_size
                self.w_ += self.eta * dw
                # print('epoch', _epoch, self.cost(batch_x, batch_y, batch_size))

    def predict(self, x):
        print(self.hypothesis(x))
        return np.where(self.hypothesis(x) >= 0.0, 1, -1)


    def cost(self, x, y, data_size):
        max_ = np.vectorize(lambda m, n: max(m, n))
        return (self.w_**2).sum()/2.0 + self.c * max_((1 - y * self.hypothesis(x)), 0).sum() / data_size

    def hypothesis(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]


class mySVM:
    def __init__(self, c=1.15, eta=0.01, max_iter=2000, random_state=1):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.c = c
    
    def fit(self, x, y):
        self.bin_classifiers = [BinarySVM(eta=self.eta, max_iter=self.max_iter, \
            random_state=self.random_state, c= self.c, label=label) for label in np.unique(y)]

        for classifier in self.bin_classifiers:
            classifier.fit(x, np.where(y == classifier.label, 1, -1))
        
    def predict(self, x):
        prediction = np.argmax([classifier.hypothesis(x) for classifier in self.bin_classifiers], axis=0)
        prediction = np.array([self.bin_classifiers[p].label for p in prediction])
        return prediction