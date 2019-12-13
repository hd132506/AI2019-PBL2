from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from skimage.feature import hog

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

def hog_descriptors(images):
    new_desc = []

    c = len(images)

    for i in images:
        new_desc.append(hog(i, orientations=8, pixels_per_cell=(4, 4),
                cells_per_block=(1, 1), visualize=True, multichannel=False)[0])
        if c % 1000 == 0:
            print(c)
        c -= 1
    return new_desc
