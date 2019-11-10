from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import random
import os
import struct
import matplotlib.pyplot as pyplot
path = './'
class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, learning_rate=0.01, max_iter = 100, batch_size = 32):
        self.C = C
        self.w = np.random.normal(0.0, 0.1, np.shape(X)[1])
        self.b = np.random.normal(0.0, 0.1, 1)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size


    def grad_wb(self, X, y, idx):
        # initialize w, b
        ws = np.zeros(len(x[0]))
        bs = 0.0
        for i in idx :
            if y[i]*(np.dot(w, X[i])+b) < 1:
                ws += -1*y[i]*np.array(X[i]) + 1/self.C*self.w
                bs += -1*y[i] + 0
            else :
                ws += 0 + 1/self.C*self.w
                bs += 0 + 0
        grad_w = ws / self.batch_size
        grad_s = bs / self.batch_size
        return self.w - learning_rate * grad_w, self.b - learning_rate * grad_b

    # def hinge(self, X, y):



    def fit(self, X, y):
        random_idx = range(np.shape(X)[0])
        random.shuffle(random_idx)
        for k in range(self.max_iter):
            batch = random_idx[k*batch_size:(k+1)*batch_size]
            self.w, self.b = grad_wb(X, y, batch)
        return self

    def predict(self, X):
        # pred = []
        # for img in X :
        #
        #     pred.append()
        # return pred
        return np.where((np.dot(X,self.w)+self.b) >= 0,1,-1)

# load data
tt_img = os.path.join(path, 't10k-images-idx3-ubyte')
tt_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
tr_img = os.path.join(path, 'train-images-idx3-ubyte')
tr_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
with open(tr_lbl, 'rb') as trlbl:
  magic, num = struct.unpack(">II", trlbl.read(8))
  train_lbl = np.fromfile(trlbl, dtype=np.int8)
with open(tr_img, 'rb') as trimg:
  magic, num, rows, cols = struct.unpack(">IIII", trimg.read(16))
  train_img = np.fromfile(trimg, dtype=np.uint8).reshape(len(train_lbl), rows * cols)
with open(tt_lbl, 'rb') as ttlbl:
  magic, num = struct.unpack(">II", ttlbl.read(8))
  test_lbl = np.fromfile(ttlbl, dtype=np.int8)
with open(tt_img, 'rb') as ttimg:
  magic, num, rows, cols = struct.unpack(">IIII", ttimg.read(16))
  test_img = np.fromfile(ttimg, dtype=np.uint8).reshape(len(test_lbl), rows * cols)

# Standardize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(train_img)
train_img = sc.transform(train_img)
test_img = sc.transform(test_img)

# Cross Validation ; K-fold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

cvalues = [0.001, 0.01, 0.1, 1.0]
param_grid = {
    'C' : [0.001, 0.01, 0.1],
    'learning_rate' : [0.01, 0.1, 1.0]}
clf = MyClassifier
gs = GridSearchCV(clf, param_grid, cv=5)
gs.fit(train_img, train_lbl)
print(gs.cv_results_)
print(gs.best_params_)
best_c = gs.best_params_['C']
best_lr = gs.best_params_['learning_rate']

sgd_svm = MyClassifier(C = best_c, learning_rate = best_lr)
sgd_svm.fit(train_img, train_lbl)

predicted = sgd_svm.predict(test_img)
expected = test_lbl

accuracy = metrics.accuracy_score(predicted, expected)
print("Classification Report for SVM\n%s:\n%s" %(svm, metrics.classification_report(expected, predicted)))
