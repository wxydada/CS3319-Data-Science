import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from Config import *


def pre_process(X=None, y=None):
    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    np.save(X_TRAIN, X_train)
    np.save(X_TEST, X_test)
    np.save(Y_TRAIN, y_train)
    np.save(Y_TEST, y_test)
    return [X_train, X_test, y_train, y_test]


def read_data():
    features = np.loadtxt(features_file)
    labels = np.loadtxt(labels_file)
    return features, labels


def run_svc(X_train, X_test, y_train, y_test, C, kernel):
    model = SVC(C=C, kernel=kernel, gamma='auto')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


def run_linear_svc(X_train, X_test, y_train, y_test, C, ):
    model = LinearSVC(C=C)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


def SVM_recommend(**SVM_paras):
    if SVM_paras == {}:
        SVM_paras = {'C': 0.01, 'max_iter': 2000}
    return LinearSVC(**SVM_paras)


if __name__ == "__main__":
    features = np.loadtxt("../../data/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt")
    labels = np.loadtxt("../../data/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=1)
    print(X_train.shape)
    print("data process complete")
    # for c in all_C:
    #     print("\nC = %f " % (c))
    #     model = SVC(kernel='linear', C=c)
    #     scores = cross_val_score(model, X_train, y_train, cv=5,n_jobs=4)
    #     print("Accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores) * 2))
    print(run_svc(X_train,X_test,y_train,y_test,0.001,"linear"))
