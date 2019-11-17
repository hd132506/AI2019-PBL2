from svm import mySVM
import numpy as np
from loader import load
from sklearn import metrics
from sklearn.decomposition import PCA
from utility import *
from sklearn.model_selection import GridSearchCV


# All data
size = 80000

X, Y = load('all', data_size=size)

shuffled_indices = np.arange(size)
np.random.shuffle(shuffled_indices)

split = 70000

X = X[shuffled_indices]
X, tx = X[:split], X[split:]

Y = Y[shuffled_indices]
Y, ty = Y[:split], Y[split:]


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
evaluation(predicted, ty)



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