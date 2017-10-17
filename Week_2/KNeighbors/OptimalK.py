import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.metrics as mt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

def Calculate(X, y):
    best_score = 0.0
    best_k = 0
    gen =  KFold(n_splits = 5, shuffle = True, random_state = 42)
    for k in range(1,50):
        #print('iteration = ', k)
        #print(X.as_matrix().shape)
        #print(y.as_matrix().ravel().shape)
        score = cross_val_score(
            estimator = KNeighborsClassifier(n_neighbors=k),
            #X = X.as_matrix(),
            #y = y.as_matrix().ravel(),
            X = X,
            y = y,
            scoring = 'accuracy',
            cv = gen)
        if (score.mean() > best_score):
            best_score = score.mean()
            best_k = k

    return best_k, best_score

data = pd.read_csv('vine.csv', header=None,)

X = data.iloc[0:,1:14]
y = data.iloc[0:,0:1]

k, sc = Calculate(X.as_matrix(), y.as_matrix().ravel())
print(k, round(sc, 2))

#print("row 39", scale(y))
k, sc = Calculate(scale(X), y.as_matrix().ravel())# scale(y).ravel())
print(k, round(sc, 2))