import numpy as np
import random



class mySVM:
    def __init__(self, c=1.15, eta=0.01, max_iter=100, random_state=1):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.c = c
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])


        data_size = len(y)

        batch_idx = [i for i in range(data_size)]
        random.shuffle(batch_idx)


        # To be parameterized
        batch_size = 4
        max_iter = 100

        
        for i in range(0, batch_size * max_iter, batch_size):
            start_idx = i % len()
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # TODO
            if batch_y * self.hypothesis(batch_x) <= 1:
                diff_w = -batch_y * batch_x
            else:
                diff_w = 0
            diff_w += self.c * self.w_





    def predict(self, x):
        return np.where(self.hypothesis(x) >= 0.0, 1, -1)

    def cost(self, x, y):
        max_ = np.vectorize(lambda m, n: max(m, n))
        return (self.w_**2)/2.0 + self.c * max_((1 - y * self.hypothesis(x)), 0) / len(y)

    def hypothesis(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]