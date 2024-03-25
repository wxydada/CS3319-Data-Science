import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from Config import *


def pre_process(X=None, y=None, bReset=False):
    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    np.save(X_TRAIN, X_train)
    np.save(X_TEST, X_test)
    np.save(Y_TRAIN, y_train)
    np.save(Y_TEST, y_test)
    return [X_train, X_test, y_train, y_test]


features = np.loadtxt(features_file)
labels = np.loadtxt(labels_file)
X_train,X_test,y_train,y_test=pre_process(features,labels)
print(X_test.shape)
print(X_train.shape)