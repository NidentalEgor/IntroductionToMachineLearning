import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.metrics as mt
import sklearn.datasets as ds
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

# boston_data = ds.load_boston()
# #print(boston_data)
# boston_X = pd.DataFrame(boston_data.data)
# scaled_boston_data = sklearn.preprocessing.scale(
#     boston_X)
# #print(boston_X)

def Calculate(X, y):
    gen =  KFold(
        n_splits = 5,
        shuffle = True,
        random_state = 42)

    best_score = -100.0
    best_k = 0.0
    for k in np.linspace(1,10,200):
        score = cross_val_score(
                    estimator = KNeighborsRegressor(
                        n_neighbors=5,
                        weights='distance',
                        p = k),
                    #X = X.as_matrix(),
                    #y = y.as_matrix().ravel(),
                    X = X,
                    y = y,
                    scoring = 'neg_mean_squared_error',
                    cv = gen)
        # print(score.mean())
        # print(round(k, 2))
        if (score.mean() > best_score):
            best_score = score.mean()
            print(score.mean())
            print(round(k, 2))
            best_k = round(k, 2)

    # print(round(best_k, 2))
    return round(best_k, 2), best_score

boston_data = ds.load_boston()
#print(boston_data)
boston_X = pd.DataFrame(boston_data.data)
scaled_boston_data = sklearn.preprocessing.scale(
    boston_X)

k, sc = Calculate(
    scaled_boston_data,
    pd.DataFrame(boston_data.target).as_matrix())
print(k,sc)