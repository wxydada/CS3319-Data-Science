import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from sklearn.decomposition import KernelPCA
sys.path.append("..")
from SVM import read_data, pre_process, run_svc, run_linear_svc, SVM_recommend

def PCA(X,y):
    scores=[]
    dimensions=[2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1200, 1500, 2000]
    for d in dimensions:
        transformer = KernelPCA(n_components=d,kernel='linear', copy_X=True,n_jobs=4)
        transformer.fit(X)
        X_ = transformer.transform(X)
        print(X_.shape)
        X_train, X_test, y_train, y_test = pre_process(X_, y)
        score_linear = run_svc(X_train, X_test, y_train, y_test, 0.001, "linear")
        print("Score : %0.4f" % (score_linear))
        scores.append(score_linear)
    plt.figure()
    plt.plot(dimensions, scores, 'bo-', linewidth=2)
    plt.title('PCA with ' + 'linear')
    plt.xlabel('dimensions')
    plt.ylabel('Score')
    plt.savefig('PCA_' + 'linear' + '.jpg')

if __name__ == '__main__':
    # X, y = read_data()
    # PCA(X,y)
    dimensions = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1200, 1500, 2000]
    scores=[0.2464,0.5601,0.7800,0.8723,0.9128,0.9228,0.9267,0.9310,0.9313,0.9319,0.9317,0.9317,0.9324]
    plt.figure()
    plt.plot(dimensions,scores,'bo-',linewidth=1,label='PCA score')
    plt.axhline(y=0.9323,label='baseline')
    plt.title('PCA with ' + 'linear')
    plt.xlabel('dimensions')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('PCA' + 'linear' + '.jpg')
