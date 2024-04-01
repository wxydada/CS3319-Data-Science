import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, chi2, SelectFromModel)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

sys.path.append("..")
from SVM import read_data, pre_process, run_svc, run_linear_svc, SVM_recommend


def Variance_Th(X, y):
    comp_range = [0.05 ,0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    dimension = []
    svc_score = []
    for var in comp_range:
        selector = VarianceThreshold(threshold=var)
        X_ = selector.fit_transform(X)
        print("Now dimension: %f"%(X_.shape[1]))
        X_train, X_test, y_train, y_test = pre_process(X_, y)
        dimension.append(X_train.shape[1])
        score_svc = run_svc(X_train, X_test, y_train, y_test, 0.001, "linear")
        svc_score.append(score_svc)
        print("Score : %0.4f"%(score_svc))
    plt.figure()
    plt.plot(dimension, svc_score, 'bo-', linewidth=1)
    plt.title('VT with ' + 'linear')
    plt.xlabel('dimensions')
    plt.ylabel('Score')
    plt.savefig('VT' + 'linear__' + '.jpg')
    return svc_score, dimension


def SelectFM(X, y):
    X_train, X_test, y_train, y_test = pre_process(X, y)
    clf = SVM_recommend()
    m_range = [2048 - 50 * i for i in range(20, 35)]
    for m in m_range:
        selector = SelectFromModel(clf, threshold=-np.inf, max_features=m)  # 只根据max_features确定选择的数量，不设定threshold
        X_ = selector.fit_transform(np.asarray(X), np.asarray(y))
        print("Now dimension: %f"%(X_.shape[1]))
        X_train, X_test, y_train, y_test = pre_process(X_, y)
        score = run_svc(X_train, X_test, y_train, y_test, 0.001, "linear")
        print("Score : %0.4f" % (score))


def SelectKbest(X,y):
    comp_range = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1200, 1500, 2000]
    linear_scores = []
    for n_comp in comp_range:
        selector = SelectKBest(score_func=chi2, k=n_comp)
        selector.fit(X, y)
        X_=selector.fit_transform(X,y)
        print("Now dimension: %f"%(X_.shape[1]))
        X_train, X_test, y_train, y_test = pre_process(X_, y)
        score_linear = run_svc(X_train, X_test, y_train, y_test, 0.001, "linear")
        linear_scores.append(score_linear)
        print("Score : %0.4f" % (score_linear))
    plt.figure()
    plt.plot(comp_range, linear_scores, 'ko-', linewidth=1)
    plt.title('SelectKbest with ' + 'linear')
    plt.xlabel('dimensions')
    plt.ylabel('Score')
    plt.savefig('SKB' + 'linear' + '.jpg')
    return linear_scores

if __name__ == "__main__":
    X, y = read_data()
    SelectKbest(X,y)
