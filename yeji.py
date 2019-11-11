from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import random
import os
import struct
import matplotlib.pyplot as pyplot
path = './'
class SGC_SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, learning_rate=0.01, max_iter = 100, batch_size = 64):
        self.C = C
        self.w = np.random.normal(0.0, 1.0, np.shape(train_img)[1])
        self.b = np.random.normal(0.0, 1.0, 1)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size


    def grad_wb(self, X, y, idx):
        # initialize w, b
        ws = np.zeros(len(X[0]))
        bs = 0.0
        for i in idx :
            if y[i]*(np.dot(self.w, X[i]) + self.b) < 1:
                ws += -1*y[i]*np.array(X[i])
                bs += -1*y[i]
            else :
                ws += 0
                bs += 0
        grad_w = ws / self.batch_size + (1/self.C)*self.w
        grad_b = bs / self.batch_size + 0
        return self.w - self.learning_rate * grad_w, self.b - self.learning_rate * grad_b

    # def hinge(self, X, y):



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
                self.w, self.b = self.grad_wb(X, y_class, batch)
                self.w_classes[dig] = self.w
                self.b_classes[dig] = self.b
        return self

    def predict(self, X):
        pred = []
        for w, b in zip(self.w_classes, self.b_classes):
            pred.append(np.dot(X, w) + b)
        ova_pred = np.argmax(pred, axis = 0)
        fin = []
        for i in range(X.shape[0]):
            fin.append(ova_pred[i])
        return fin

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

param_grid = {
    'C' : [0.001, 0.01, 0.1],
    'learning_rate' : [0.01, 0.1, 1.0]}
clf = SGC_SVM()
gs = GridSearchCV(clf, param_grid, cv=10)
gs.fit(train_img, train_lbl)
print(gs.cv_results_)
print(gs.best_params_)
best_c = gs.best_params_['C']
best_lr = gs.best_params_['learning_rate']
print(best_c, best_lr)

sgd_svm = SGC_SVM(C = best_c, learning_rate = best_lr)
sgd_svm.fit(train_img, train_lbl)

predicted = sgd_svm.predict(test_img)
print(predicted)
expected = test_lbl

accuracy = metrics.accuracy_score(predicted, expected)
print("Classification Report for SVM\n%s:\n%s" %(sgd_svm, metrics.classification_report(expected, predicted)))
