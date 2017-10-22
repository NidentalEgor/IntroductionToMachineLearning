import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import time
import datetime

def ReadData():
    features = pd.read_csv('./features.csv', index_col='match_id')
    radiant_win = features['radiant_win']

    return features, radiant_win



def DropMatchResultFutures(features):
    # Removing match result futures.
    features = features.drop(
        ['barracks_status_dire',
            'barracks_status_radiant',
            'tower_status_dire',
            'tower_status_radiant',
            'radiant_win',
            'duration'],
        axis=1)
    
    columns_with_missings = features.shape[0] - features.count(axis=0)
    
    # Columns with missings.
    print(columns_with_missings[columns_with_missings > 0])

    features.fillna(0, inplace = True)

    return features



def PrepareDataForGradientBoosting():
    features, radiant_win = ReadData()

    return DropMatchResultFutures(features), radiant_win


def ReadTestdData():
    features = pd.read_csv('./features_test.csv', index_col='match_id')
    features.fillna(0, inplace = True)

    return features



def GradientBoostingCrossValidation(X_train, y_train):
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



def DropCategorialFeatures(features):
    features = features.drop(
        ['lobby_type',
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero'],
        axis=1)

    return features



def RegressionCrossValidation(X_train, y_train):
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



def GetHeroesAmount(X):
    heroes_values = np.unique(X[['r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero']].values)

    print("Heroes =", heroes_values)
    print("Heroes amount =", heroes_values.shape[0])
    return heroes_values.shape[0]



def ReplaceCategorialFeaturesWithBagOfWords(X, N):
    categorial_features_names = [
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero']
    categorial_features = X[categorial_features_names]

    bag_of_words = np.zeros((X.shape[0], N))

    for i, match_id in enumerate(X.index):
        for p in range(5):
            bag_of_words[i, X.ix[match_id, 'r%d_hero' % (p+1)]-5] = 1
            bag_of_words[i, X.ix[match_id, 'd%d_hero' % (p+1)]-5] = -1

    X = X.drop(
            categorial_features_names,
            axis=1)
    result = pd.DataFrame(np.concatenate((X.as_matrix(), bag_of_words), axis=1))

    return X



def GetFinalPrediction(X_train, y_train, X_test):
    logistic_regression = LogisticRegression(
            C=0.01,
            random_state = 241)
    logistic_regression.fit(X_train, y_train)
    result = logistic_regression.predict_proba(X_test)

    return result[:,1]



def DoGradientBoosting():
    data, target_variable = PrepareDataForGradientBoosting()
    GradientBoostingCrossValidation(data, target_variable)



def DoLogisticRegression():
    X_train_original, y = ReadData()
    X_train_without_match_result_features = DropMatchResultFutures(
            X_train_original)
    scaler = StandardScaler()

    # X_train_with_categorial_features_scaled = scaler.fit_transform(
    #       X_train_without_match_result_features.as_matrix())

    # RegressionCrossValidation(
    #       X_train_with_categorial_features_scaled,
    #       y)
    # Best score = 0.653792271116 , best c = 10.0

    X_train_without_categorial_features = DropCategorialFeatures(
            X_train_without_match_result_features)
    X_train_without_categorial_features_scaled = scaler.fit_transform(
            X_train_without_categorial_features.as_matrix())

    # RegressionCrossValidation(
    #       X_train_without_categorial_features_scaled,
    #       y)
    # Best score = 0.65396795402 , best c = 0.01

    
    # X_train_with_bag_of_words = ReplaceCategorialFeaturesWithBagOfWords(
    #       X_train_without_match_result_features,
    #       GetHeroesAmount(X_train_without_match_result_features))

    # X_train_with_bag_of_words_scaled = scaler.fit_transform(
    #         X_train_with_bag_of_words.as_matrix())

    # RegressionCrossValidation(
    #         X_train_with_bag_of_words_scaled,
    #         y)
    # Best score = 0.653813650883 , best c = 0.01

    X_test = ReadTestdData()
    X_test_without_categorial_features = DropCategorialFeatures(
            X_test)
    X_test_without_categorial_features_scaled = scaler.fit_transform(
            X_test_without_categorial_features.as_matrix())

    result = GetFinalPrediction(
            X_train_without_categorial_features_scaled,
            y,
            X_test_without_categorial_features_scaled)

    print("Min predict result =", result.min())
    print("Max predict result =", result.max())

    match_id = X_test.index.values
    result = result.reshape((result.shape[0],1))
    match_id = np.around(match_id.reshape((result.shape[0],1)),decimals = 1)
    final_result = np.concatenate(
            (match_id,
            result),
            axis = 1)

    np.savetxt('result.csv', final_result, fmt='%d, %lf')



def main():
    ######### Gradient boosting. #########
    DoGradientBoosting()

    ######### Logistic regression. #########
    DoLogisticRegression()

if __name__ == "__main__":
    main()