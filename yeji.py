from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import random
import os
import sys
import struct
import matplotlib.pyplot as pyplot
path = './'
class SGC_SVM(BaseEstimator, ClassifierMixin):
    # C = 1/lambda , eta = learning_rate
    def __init__(self, C=1.0, eta=0.01, max_iter = 300, batch_size = 32):
        self.C = C
        self.w = np.random.normal(0.0, 1.0, np.shape(train_img)[1])
        self.b = np.random.normal(0.0, 1.0, 1)
        self.eta = eta
        self.max_iter = max_iter
        self.batch_size = batch_size

    def grad(self, X, y, idx):
        # initialize w, b
        ws = np.zeros(len(X[0]))
        bs = 0.0
        for i in idx :
            if y[i]*(np.dot(self.w, X[i]) + self.b) < 1:
                ws += -1*y[i]*np.array(X[i], dtype=np.float64) + (1/self.C)*self.w
                bs += -1*y[i] + 0
            else :
                ws += 0 + (1/self.C)*self.w
                bs += 0 + 0
        grad_w = ws / self.batch_size
        grad_b = bs / self.batch_size
        return self.w - self.eta * grad_w, self.b - self.eta * grad_b

    def fit(self, X, y):
        self.w_classes = [self.w for _ in range(10)]
        self.b_classes = [self.b for _ in range(10)]
        random_idx = list(range(np.shape(X)[0]))
        random.shuffle(random_idx)
        for dig in range(10):
            self.w = self.w_classes[dig]
            self.b = self.b_classes[dig]
            y_class = [1 if yi == dig else -1 for yi in y]
            for k in range(self.max_iter):
                batch = random_idx[k*self.batch_size:(k+1)*self.batch_size]
                self.w, self.b = self.grad(X, y_class, batch)
                self.w_classes[dig] = self.w
                self.b_classes[dig] = self.b
        return self

    def predict(self, X):
        pred = []
        for w, b in zip(self.w_classes, self.b_classes):
            pred.append(np.dot(X,w) + b)
        print(pred[0])
        pred_class = np.argmax(pred, axis = 0)
        pred_result = []
        f = open('result.txt', mode='w')
        for i in range(X.shape[0]):
            f.write(str(pred_class[i])+'\n')
            pred_result.append(pred_class[i])
        f.close()
        return pred_result

# load data
tt_img = os.path.join(path, 'test-images-idx3-ubyte')
tt_lbl = os.path.join(path, 'test-labels-idx1-ubyte')
tr_img = os.path.join(path, 'newtrain-images-idx3-ubyte')
tr_lbl = os.path.join(path, 'newtrain-labels-idx1-ubyte')
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

param_grid = {
    'C' : [0.01, 0.05, 0.1, 0.5, 1.0],
    'eta' : [0.001, 0.01, 0.1]}
clf = SGC_SVM()
gs = GridSearchCV(clf, param_grid, cv=10)
gs.fit(train_img, train_lbl)
best_c = gs.best_params_['C']
best_eta = gs.best_params_['eta']
print("Best C value : ",best_c,"\nBest eta value : " ,best_eta)

sgd_svm = SGC_SVM(C = best_c, eta = best_eta)
sgd_svm.fit(train_img, train_lbl)

predicted = sgd_svm.predict(test_img)
expected = test_lbl

accuracy = metrics.accuracy_score(predicted, expected)
print("Classification Report for SVM\n%s:\n%s" %(sgd_svm, metrics.classification_report(expected, predicted)))
