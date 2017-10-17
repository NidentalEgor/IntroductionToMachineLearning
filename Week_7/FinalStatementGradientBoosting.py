import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
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

def GradientBoosting(X_train, y_train):
    kfold =  KFold(
        n_splits = 5,
        shuffle = True,
        random_state = 241)
    
    iter = 1
    best_trees_amount = 0.000001
    max_score = 0.
    for trees_amount in [10, 20, 30, 31, 33, 35, 37, 39]:
        start_time = datetime.datetime.now()

        score = cross_val_score(
                estimator = GradientBoostingClassifier(
                    n_estimators=trees_amount,
                    verbose=True,
                    random_state=241,
                    learning_rate=0.5),
                X = X_train,
                y = y_train,
                scoring = make_scorer(roc_auc_score),
                cv = kfold)
        
        print("Trees amount =",trees_amount,"Time elapsed:", datetime.datetime.now() - start_time)
        print("\nIter =",iter,"score =",score.mean(),"\n")

        if score.mean() > max_score:
            best_trees_amount = trees_amount
            max_score = score.mean()

        iter+=1

    print("Best score =",max_score,", best best_trees_amount =",best_trees_amount)
    # Best score = 0.644067560767 , best best_trees_amount = 39  

def main():
    data, target_variable = PrepareData()
    # Target variable - radiant_win.

    GradientBoosting(data, target_variable)


if __name__ == "__main__":
    main()