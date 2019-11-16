from svm import mySVM
import numpy as np
from loader import load
from sklearn import metrics
from sklearn.decomposition import PCA
from utility import *
from sklearn.model_selection import GridSearchCV


size = 80000

X, Y = load('all', data_size=size)

shuffled_indices = np.arange(size)
np.random.shuffle(shuffled_indices)

split = 15000

X = X[shuffled_indices]
X, tx = X[:split], X[split:]

Y = Y[shuffled_indices]
Y, ty = Y[:split], Y[split:]





X = scale(X)
tx = scale(tx)

# PCA transformation
pca = PCA(.95)
pca.fit(X)

X = pca.transform(X)
tx = pca.transform(tx)


# model = mySVM(c=0.9, eta=0.0001, forget_factor=0.9, max_iter=1000)
# model.fit(X, Y)

# predicted = model.predict(tx)
# evaluation(predicted, ty)


hyper_params = {"c": [0.1, 112.5, 125, 137.5, 150, 175], \
    "eta": [0.0001, 0.005, 0.01, 0.02, 0.1], \
    "forget_factor": [0.001, 0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]}
gs = GridSearchCV(mySVM(), hyper_params, n_jobs=8)

gs.fit(X, Y)

print(gs.best_params_)

c, eta, f = gs.best_params_['c'], gs.best_params_['eta'], gs.best_params_['forget_factor']
model = mySVM(c=c, eta=eta, forget_factor=f, max_iter=1000)
model.fit(X, Y)
predicted = model.predict(tx)
evaluation(predicted, ty)