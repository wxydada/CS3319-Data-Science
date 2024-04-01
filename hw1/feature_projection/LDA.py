import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sys.path.append("..")
from SVM import read_data, pre_process, run_svc, run_linear_svc, SVM_recommend

def LDA(X,y):
    scores=[]
    dimensions=[2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1200, 1500, 2000]
    for d in dimensions:
        transformer = LinearDiscriminantAnalysis(n_components=d)
        transformer.fit(X,y)
        X_ = transformer.transform(X)
        print(X_.shape)
        X_train, X_test, y_train, y_test = pre_process(X_, y)
        score_linear = run_svc(X_train, X_test, y_train, y_test, 0.001, "linear")
        print("Score : %0.4f" % (score_linear))
        scores.append(score_linear)
    plt.figure()
    plt.plot(dimensions, scores, 'bo-', linewidth=2)
    plt.title('LDA with ' + 'linear')
    plt.xlabel('dimensions')
    plt.ylabel('Score')
    plt.savefig('LDA_' + 'linear' + '.jpg')
    return scores
if __name__ == '__main__':
    X, y = read_data()
    LDA(X,y)