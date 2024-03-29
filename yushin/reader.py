import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset="training", data_size=10000, path="./dataset"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "testing":
        data_size = min(data_size, 10000)
        fname_img = os.path.join(path, 'mnist_new_test-patterns-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist_new_test-labels-idx1-ubyte')
    elif dataset is "training":
        data_size = min(data_size, 60000)
        fname_img = os.path.join(path, 'mnist_new-patterns-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist_new-labels-idx1-ubyte')
    elif dataset is "new":
        data_size = min(data_size, 10000)
        fname_img = os.path.join(path, 'mnist-new1k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist-new1k-labels-idx1-ubyte')
    elif dataset is "all": 
        data_size = min(data_size, 80000)
        fname_img = os.path.join(path, 'newtrain-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'newtrain-labels-idx1-ubyte')
    else:
        raise Exception("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _magic, _num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _magic, _num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(data_size):
        yield get_img(i)
