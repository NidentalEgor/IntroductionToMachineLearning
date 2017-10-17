import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import time
import datetime

def PrepareData():
    features = pd.read_csv('./features.csv', index_col='match_id')
    
    radiant_win = features['radiant_win']

    # Removing data from future.
    features = features.drop(
        ['barracks_status_dire',
            'barracks_status_radiant',
            'tower_status_dire',
            'tower_status_radiant',
            'radiant_win',
            'duration'],
        axis=1)
    
    ###
    # print(features.head())
    # print(features.columns.tolist())
    ###
    
    columns_with_missings = features.shape[0] - features.count(axis=0)
    # This is columns with missings.
    print(columns_with_missings[columns_with_missings > 0])

    # TODO: try to replace with large value.
    features.fillna(0, inplace = True)

    return features, radiant_win

def DoRegression(X_train, y_train):
    kfold =  KFold(
        n_splits = 5,
        shuffle = True,
        random_state = 241)

    iter = 1
    best_c = 0.000001
    max_score = 0.
    for c in np.power(10.0, np.arange(-5, 6)):
        start_time = datetime.datetime.now()
        score = cross_val_score(
                estimator = LogisticRegression(
                    C=c,
                    random_state = 241),
                X=X_train,
                y=y_train,
                scoring=make_scorer(roc_auc_score),
                cv=kfold)

        print("\nC =", c,"Time elapsed:",datetime.datetime.now() - start_time)
        print("\nIter =",iter,"score =",score.mean(),"\n")

        if score.mean() > max_score:
            best_c = c
            max_score = score.mean()

        iter+=1
    
    print("Best score =",max_score,", best c =",best_c)
    # Best score = 0.653792271116 , best c = 10.0

def main():
    X, y = PrepareData()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.as_matrix())
    print(X_scaled)

    DoRegression(X_scaled, y)


if __name__ == "__main__":
    main()