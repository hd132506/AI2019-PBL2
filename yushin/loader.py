import reader
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Retun actually trainable/testable dataset using reader function
dta_size: Number of training data. Maximum: 10,000
"""
def load(dataset="training", data_size=10000):

    # Lamda functins for preprocessing
    imgs = lambda dset : \
        np.array([data[1] for data in dset], dtype='f')
    labels = lambda dset : np.array([data[0] for data in dset])

    dset = list(reader.read(dataset=dataset, data_size=data_size))



    return (imgs(dset), labels(dset))