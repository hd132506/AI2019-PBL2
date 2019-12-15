import os
import struct
import numpy as np
import sys

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
        
        ans = -np.ones(scores.shape)
        ans[np.arange(data_size), y] = 1

        scores *= ans
        scores = np.where(scores < 1, -1., 0)
        scores *= ans

        dw = np.dot(x.T, scores) / data_size

        dw[1:] += (1. / self.c) * self.w_[1:]

        return dw

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)

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

                # # Naive SGD
                # self.w_ -= self.eta*dw

                # NADAM
                m = self.mu*m + (1-self.mu)*dw
                n = self.v*n + (1-self.v)*np.power(dw, 2)
                m_hat = (self.mu*m/(1 - mu_pw*self.mu)) + ((1-self.mu)*dw/(1-mu_pw))
                n_hat = self.v*n / (1-v_pw)
                self.w_ -= self.eta * m_hat / np.sqrt(epsilon + n_hat)

                mu_pw *= self.mu
                v_pw *= self.v

        return self
        
    def predict(self, x):
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        return np.argmax(np.dot(x, self.w_), axis=1)

    def score(self, x, y):
        err = y - self.predict(x)
        err = np.where(err == 0, 1, 0)
        return sum(err)
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(image, lbl='', path="./"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    fname_img = os.path.join(path, image)
    img_lbl = lambda idx: (img[idx], lbl[idx])

    with open(fname_img, 'rb') as fimg:
        _magic, _num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(_num, rows, cols)


    if lbl is not '':
        print(image)
        fname_lbl = os.path.join(path, lbl)
        with open(fname_lbl, 'rb') as flbl:
            _magic, _num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        for i in range(_num):
            yield img_lbl(i)
        return

    get_img = lambda idx: (img[idx], 0)
    for i in range(_num):
        yield get_img(i)

    # Create an iterator which returns each image in turn
    





"""
Retun actually trainable/testable dataset using reader function
dta_size: Number of training data. Maximum: 10,000
"""
def load(image, label=''):

    # Lamda functins for preprocessing
    imgs = lambda dset : \
        np.array([data[0] for data in dset], dtype='f')
    flatten_labels = lambda dset : np.array([data[1] for data in dset])

    dset = list(read(image=image, lbl=label))

    if label is '':
        return imgs(dset)
    return (imgs(dset), flatten_labels(dset))

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler



# Utility functions 

def flatten(images):
    return images.reshape((-1, 28*28))

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



def hog(img, cell_size=4, block_size=2, orientations=8):
    """
    Histogram of Gradient operation
    Using square-shaped cell and block
    img: 2D numpy array
    cell_size: the number of pixels in one side of a cell
    block_size: the number of cells in one side of a block
    orientations: the number of unsigned orientations
    """
    # Get gradients
    y_gradient = np.zeros(img.shape)
    x_gradient = np.zeros(img.shape)
    y_gradient[1:-1, :] = img[2:, :] - img[:-2, :]
    x_gradient[:, 1:-1] = img[:, 2:] - img[:, :-2]

    # Compute magnitude and orientation matrice
    magnitude_mat = np.sqrt(x_gradient**2 + y_gradient**2)
    orientation_mat = np.rad2deg(np.arctan(y_gradient/(x_gradient + 10**(-8)))) % 180

    # Quantitize orientations
    ori_arr = np.array([(i*180/orientations) for i in range(orientations)])
    quant = lambda d: np.argmin(np.minimum(np.abs(d - ori_arr), 180 - np.abs(d - ori_arr)))

    # For transforming pixel position to cell position
    cell_pos = lambda k: k//cell_size

    # Make orientation histogram for each cell
    cell_hist = np.zeros((cell_pos(img.shape[0]), cell_pos(img.shape[1]), orientations))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            cell_hist[cell_pos(y), cell_pos(x), quant(orientation_mat[y, x])] \
            += magnitude_mat[y, x]
    cell_hist /= cell_size**2

    n_cells_y = cell_pos(img.shape[0])
    n_cells_x = cell_pos(img.shape[1])

    # Block normalization
    n_blocks_y = (n_cells_y - block_size) + 1
    n_blocks_x = (n_cells_x - block_size) + 1

    
    block_normalize = lambda block: \
                    block / np.sqrt(np.sum(block ** 2) + 1e-12) # L2 Norm

    normalized_blocks = np.zeros((n_blocks_y*n_blocks_x, block_size, block_size, orientations))

    
    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            normalized_blocks[y*n_blocks_x + x, :] = block_normalize(cell_hist[y:y+block_size, x:x+block_size, :])
    
    return normalized_blocks.reshape(-1)

def hog_descriptors(images):
    new_desc = []

    for i in images:
        new_desc.append(hog(i))
    return new_desc






from sklearn.decomposition import PCA


# All data
# size = 80000
X, Y = load(image=sys.argv[1], label=sys.argv[2])
tx = load(image=sys.argv[3])


desc_X = np.array(hog_descriptors(X))
X = flatten(X)
X = np.append(scale(X), scale(desc_X), axis=1)

desc_tx = np.array(hog_descriptors(tx))
tx = flatten(tx)
tx = np.append(scale(tx), scale(desc_tx), axis=1)





model = mySVM(c=200.15, eta=0.001, mu=0.9999, v=0.9999, max_iter=37000)
model.fit(X, Y)

predicted = model.predict(tx)
np.savetxt('prediction.txt', predicted, fmt = "%1d",delimiter='\n')