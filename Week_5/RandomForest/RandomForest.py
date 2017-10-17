from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

def GetScore(X, y, tree_amount):
    cv = KFold(
        n_splits = 5,
        shuffle = True,
        random_state = 1)

        # rfr = RandomForestRegressor(
        #     random_state=1,
        #     n_estimators=tree_amount)
        # rfr.fir(X, y)

    # scorer = r2_score()

    score = cross_val_score(
            estimator = RandomForestRegressor(
                random_state=1,
                n_estimators=tree_amount),
            X = X,
            y = y,
            scoring = make_scorer(r2_score),
            cv = cv)
    
    print("score =",score)

    return score.mean()


def main():
    data = pd.read_csv("abalone.csv")
    # print(data.head())
    data['Sex'] = data['Sex'].map(
        lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    # print(data.head())
    # print(type(data['Sex']))

    X = data.iloc[0:,0:8]
    y = data.iloc[0:,8:9]

    for tree_amount in range(1,51):
        print(
            "Tree amount =",
            tree_amount,
            "Score =",
            GetScore(
                X.as_matrix(),
                y.as_matrix().ravel(),
                tree_amount))

if __name__ == "__main__":
    main()