from sklearn import metrics
import numpy as np
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