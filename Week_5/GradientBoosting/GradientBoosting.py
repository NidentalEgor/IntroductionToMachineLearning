from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
# %matplotlib inline

def GradientBoosting(X_train, X_test, y_train, y_test):
    for lr in  [0.2]:#[1, 0.5, 0.3, 0.2, 0.1]:
        print("lr =",lr)
        gbc = GradientBoostingClassifier(
            n_estimators=250,
            verbose=True,
            random_state=241,
            learning_rate=lr)
        
        gbc.fit(X_train, y_train)

        # train_loss = []
        # for i,y_pred in enumerate(gbc.staged_decision_function(X_train)):
        #     y_p = 1 / (1 + np.exp(-y_pred))
        #     train_loss.append(log_loss(y_train, y_p))

        test_loss = []
        min = 100000000.0
        iter = 0
        for i,y_pred in enumerate(gbc.staged_decision_function(X_test)):
            y_p = 1 / (1 + np.exp(-y_pred))
            ll = log_loss(y_test, y_p)
            if ll < min:
                iter = i
                min = ll        
            test_loss.append(ll)

        print("Iter =",iter,"min log_loss =", min)

        plt.figure()
        plt.plot(test_loss, 'r', linewidth=2)
        # plt.plot(train_loss, 'g', linewidth=2)
        plt.legend(['test', 'train'])
        plt.show()

def RandomForest(X_train, X_test, y_train, y_test):
    rfr = RandomForestClassifier(
                random_state=241,
                n_estimators=36)
    rfr.fit(X_train, y_train)
    r = rfr.predict_proba(X_test)
    res = log_loss(y_test, r)
    print(res)

def main():
    data = pd.read_csv("gbm-data.csv")
    # print(data.head())
    X = data.iloc[0:,1:].values
    y = data.iloc[0:,0:1].values

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X,
            y,
            test_size=0.8,
            random_state=241)

    # GradientBoosting(X_train, X_test, y_train, y_test)
    RandomForest(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()