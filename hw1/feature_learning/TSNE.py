import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import TSNE
sys.path.append("..")
from SVM import read_data, pre_process, run_svc, run_linear_svc, SVM_recommend

def T_SNE(X,y):
    scores=[]
    ppl=[10,20,30,40,50]
    dimensions = [2, 3]
    i=1
    plt.figure(figsize=(25,10))
    for d in dimensions:
        for p in ppl:
            transformer = TSNE(n_components=d,perplexity=p,method='barnes_hut', n_jobs=4,init='pca',learning_rate='auto')
            transformer.fit(X)
            X_ = transformer.fit_transform(X)
            print(X_.shape)
            print('ppl:{}'.format(p))
            X_train, X_test, y_train, y_test = pre_process(X_, y)
            score_linear = run_svc(X_train, X_test, y_train, y_test, 0.001, "linear")
            print("Score : %0.4f" % (score_linear))
            scores.append(score_linear)
            plt.subplot(2, 5, i)
            plt.scatter(X_[:, 0], X_[:, 1], c=y.astype(int))
            plt.title('{}_d_{}_ppl score:{}'.format(d,p,score_linear))
            i+=1

    plt.savefig('T_SNE.jpg')
    return scores
if __name__ == '__main__':
    X, y = read_data()
    T_SNE(X,y)
