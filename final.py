import os
import struct
import numpy as np
import sys

import random
from sklearn.base import BaseEstimator, ClassifierMixin

class BinarySVM:
    def __init__(self, c=1.15, eta=0.01, forget_factor=0.1, mu = 0.975, v = 0.999,
        max_iter=2000, random_state=1, label=-1):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.c = c
        self.forget_factor = forget_factor
        self.mu = mu
        self.v = v
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
        epsilon = 10**(-8)

        # To be parameterized
        batch_size = 128

        n_epochs = batch_size * self.max_iter // self.data_size
        n_batches = self.data_size // batch_size


        # RMS init
        # rms = epsilon

        # NADAM init
        m = 0
        n = 0
        mu_pw = self.mu
        v_pw = self.v

        for epoch in range(n_epochs):
            random_indices = BinarySVM.shuffle(self.data_size)

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
                dw /= batch_size
                dw[1:] += (1. / self.c) * self.w_[1:]


                ###  Update
                # 1. Naive SGD
                # self.w_ -= self.eta * dw

                # 2. RMSProp
                # rms = self.forget_factor * rms + (1 - self.forget_factor) * (dw**2)
                # self.w_ -= self.eta * dw / np.sqrt(epsilon + rms)

                # # 3. NADAM
                m = self.mu*m + (1-self.mu)*dw
                n = self.v*n + (1-self.v)*np.power(dw, 2)
                m_hat = (self.mu*m/(1 - mu_pw*self.mu)) + ((1-self.mu)*dw/(1-mu_pw))
                n_hat = self.v*n / (1-v_pw)
                self.w_ -= self.eta * m_hat / np.sqrt(epsilon + n_hat)

                mu_pw *= self.mu
                v_pw *= self.v



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
    def __init__(self, c=1.15, eta=0.01, forget_factor=0.1, mu = 0.975, v = 0.999,
    max_iter=3000, random_state=1):
        self.c = c
        self.eta = eta
        self.forget_factor = forget_factor
        self.mu = mu
        self.v = v
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, x, y):
        self.bin_classifiers = [BinarySVM(c= self.c, eta=self.eta, forget_factor=self.forget_factor,
        mu = self.mu, v = self.v,
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

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(path="./", img, lbl=''):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    fname_img = os.path.join(path, train_img)
    with open(fname_img, 'rb') as fimg:
        _magic, _num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(_num, rows, cols)

    if lbl != '':
        fname_lbl = os.path.join(path, train_lbl)
        with open(fname_lbl, 'rb') as flbl:
            _magic, _num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

    # if dataset is "testing":
    #     data_size = min(data_size, 10000)
    #     fname_img = os.path.join(path, 'mnist_new_test-patterns-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'mnist_new_test-labels-idx1-ubyte')
    # elif dataset is "training":
    #     data_size = min(data_size, 60000)
    #     fname_img = os.path.join(path, 'mnist_new-patterns-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'mnist_new-labels-idx1-ubyte')
    # elif dataset is "new":
    #     data_size = min(data_size, 10000)
    #     fname_img = os.path.join(path, 'mnist-new1k-images-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'mnist-new1k-labels-idx1-ubyte')
    # elif dataset is "all":
    #     data_size = min(data_size, 80000)
    #     fname_img = os.path.join(path, 'newtrain-images-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'newtrain-labels-idx1-ubyte')
    # else:
    #     raise Exception("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays



    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(data_size):
        yield get_img(i)

# import reader
# import numpy as np
# from sklearn.preprocessing import StandardScaler

"""
Retun actually trainable/testable dataset using reader function
dta_size: Number of training data. Maximum: 10,000
"""
def load(image, label=''):

    # Lamda functins for preprocessing
    flatten_imgs = lambda dset : \
        np.array([data[1].reshape(-1) for data in dset], dtype='f')
    flatten_labels = lambda dset : np.array([data[0] for data in dset])

    dset = list(read(img=image, lbl=label))


    return (flatten_imgs(dset), flatten_labels(dset))

from sklearn import metrics
# import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

def evaluation(prediction, goal):
    print("Classification report\n%s\n"
        % (metrics.classification_report(goal, prediction, digits=3)))
    print("Confusion matrix:\n%s"
        % metrics.confusion_matrix(goal, prediction))

def scale(dataset):
    # ztransform = lambda arr : (arr - np.mean(arr))/np.std(arr)
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()

    scaler.fit(dataset)
    return scaler.transform(dataset)

def binarization(dataset):
    ret = []
    for data in dataset:
        threshold = np.mean(data)
        ret.append(np.where(data < threshold, 0, 255))
    return np.array(ret)


# from svm import mySVM
# import numpy as np
# from loader import load
# from sklearn import metrics
from sklearn.decomposition import PCA
# from utility import *
from sklearn.model_selection import GridSearchCV


# All data
# size = 80000

X = load(image=sys.argv[1], label=sys.argv[2])
Y = load(image=sys.argv[3])

# shuffled_indices = np.arange(size)
# np.random.shuffle(shuffled_indices)
#
# split = 70000
#
# X = X[shuffled_indices]
# X, tx = X[:split], X[split:]
#
# Y = Y[shuffled_indices]
# Y, ty = Y[:split], Y[split:]


# # Training / Testing
# size = 60000

# X, Y = load('training', data_size=size)
# tx, ty = load('testing', data_size=10000)


# PCA transformation
pca = PCA(.95)
pca.fit(X)

X = pca.transform(X)
tx = pca.transform(tx)

X = scale(X)
tx = scale(tx)



model = mySVM(c=1000, eta=0.005, mu=0.999, v=0.99, max_iter=7000)
model.fit(X, Y)

predicted = model.predict(tx)
np.savetxt('prediction.txt', predicted, fmt = "%1d",delimiter='\n')
# evaluation(predicted, ty)



# hyper_params = {"c": [375, 400, 425, 450, 475, 500, 525, 550], \
#     "eta": [0.001, 0.002, 0.003,0.0035, 0.004, 0.0045, 0.005, 0.007, 0.01], \
#     "forget_factor": [0.1, 0.8, 0.9, 0.99, 0.999, 0.9999]}

# hyper_params_nadam = {"c": [0.8, 1.15, 100, 500, 700], \
#     "eta": [0.001, 0.002, 0.005, 0.007, 0.01, 0.02], \
#     "mu": [0.975, 0.99, 0.999, 0.9999],
#     "v": [0.975, 0.99, 0.999, 0.9999]}

# gs = GridSearchCV(mySVM(), hyper_params_nadam, n_jobs=9)

# gs.fit(X, Y)

# print(gs.best_params_)

# c, eta, mu, v = gs.best_params_['c'], gs.best_params_['eta'], gs.best_params_['mu'], gs.best_params_['v']
# model = mySVM(c=c, eta=eta, mu=mu, v=v, max_iter=10000)
# model.fit(X, Y)
# predicted = model.predict(tx)
# evaluation(predicted, ty)
